import click
import os
import jinja2
import inspect
import fnmatch
import subprocess
import yaml
import itertools
import numpy as np
import matplotlib.pyplot as mplt
import ret.plot as plt

from functools import partial
from concurrent.futures import ThreadPoolExecutor

@click.group()
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)
    # Read and parse retconfig.yml from current directory
    with open("retconfig.yml", 'r') as configfile:
        try:
            ctx.obj['config'] = yaml.safe_load(configfile)
        except yaml.YAMLError as err:
            print(err)
            print("Error in parsing retconfig.yml")
            exit(0)

def run_script(script, arguments, capture_output=False):
    # Run hook and retrieve return value
    command = [script]
    command.extend(arguments)
    completed_process = subprocess.run(command, capture_output=capture_output)
    returncode = completed_process.returncode
    if returncode != 0:
        print(f"{script} failed with error code {returncode}")
        exit(0)
    if capture_output:
        return str(completed_process.stdout, 'UTF-8')

def run_hook(config, hook_name, arguments, capture_output=False):
    # Check if hook exists in config and in filesystem
    if hook_name not in config['hooks']:
        return
    script = config['hooks'][hook_name]
    if not os.path.exists(script):
        print(f"Script {script} does not exist")
        exit(0)
    return run_script(script, arguments, capture_output)

def execute_run(benchmark, config, model, data_dir):
    # Create a folder for this run
    run_dir = os.path.join(data_dir, model, benchmark)
    if os.path.isdir(run_dir):
        print (f"{run_dir} already exists. Skipping run")
        return
    os.makedirs(run_dir)
    # Run hooks
    arguments = [model, benchmark, run_dir]
    run_hook(config, 'pre_run', arguments)
    run_hook(config, 'run', arguments)
    run_hook(config, 'post_run', arguments)

def execute_models_in_parallel(benchmarks, config, model, data_dir):
    for benchmark in benchmarks:
        execute_run(benchmark, config, model, data_dir)

def execute_benchmarks_in_parallel(benchmark, config, models, data_dir):
    for model in models:
        execute_run(benchmark, config, model, data_dir)

def execute_in_parallel(model_benchmark, config, data_dir):
    model, benchmark = model_benchmark
    execute_run(benchmark, config, model, data_dir)

def process_glob_patterns(full_list, comma_separated_list):
    data = comma_separated_list.split(",")
    new_data = []
    for item in data:
        for match_item in fnmatch.filter(full_list, item):
            if match_item not in new_data:
                new_data.append(match_item)
    for item in new_data:
        print(f"{item} ", end='')
    print()
    return list(new_data)

@cli.command()
@click.option("--benchmarks", "-b", help="Comma separated list of benchmarks to run")
@click.option("--models", "-m", required=True, help="Comma separated list of models to run")
@click.option("-j", help="Maximum number of runs to execute in parallel")
@click.pass_context
def run(ctx, benchmarks, models, j):
    config = ctx.obj['config']
    data_dir = config['data_dir']

    print("Benchmarks to run: ",end='')
    if not benchmarks:
        benchmarks = "*"
    benchmarks = process_glob_patterns(config['benchmarks'], benchmarks)

    model_names = models
    print("Models to run: ",end='')
    models = process_glob_patterns(config['models'], models)

    run_hook(config, 'pre_batch', [model_names, data_dir])

    # Create a ThreadPoolExecutor to run in parallel
    if config['run_contraint'] != 'serial':
        if j:
            j = int(j)
            executor = ThreadPoolExecutor(max_workers=j)
        else:
            executor = ThreadPoolExecutor()
    print(executor)

    if config['run_contraint'] == 'serial':
        for model in models:
            for benchmark in benchmarks:
                execute_run(benchmark, config, model, data_dir)
    elif config['run_contraint'] == 'models_in_parallel':
        run_function = partial(execute_models_in_parallel, config=config, benchmarks=benchmarks, data_dir=data_dir)
        list(executor.map(run_function, models))
    elif config['run_contraint'] == 'benchmarks_in_parallel':
        run_function = partial(execute_benchmarks_in_parallel, config=config, models=models, data_dir=data_dir)
        list(executor.map(run_function, benchmarks))
    elif config['run_contraint'] == 'parallel':
        runs = [(model, benchmark) for model in models for benchmark in benchmarks]
        run_function = partial(execute_in_parallel, config=config, data_dir=data_dir)
        list(executor.map(run_function, runs))

    run_hook(config, 'post_batch', [model_names, data_dir])

def get_model_benchmark_data(mb_tuple, data_dir, script, metric):
    model, benchmark = mb_tuple
    run_dir = os.path.join(data_dir, model, benchmark)
    data = run_script(script, [model, benchmark, run_dir, metric], capture_output=True)
    return data

def merge_dicts(list_of_dicts):
    new_dict = {}
    for dictionary in list_of_dicts:
        for item in dictionary:
            new_dict[item] = dictionary[item]
    return new_dict

def generate_plot_script(models, benchmarks, metric_name, config, savefig, filename):
    environment = jinja2.Environment(loader = jinja2.PackageLoader("ret"))
    template = environment.get_template("plot_script.py")

    # Generate plot config
    if 'default_plot_config' in config:
        plot_config = merge_dicts([plt.default_plot_config, config['default_plot_config'], config['metrics'][metric_name]])
    else:
        plot_config = merge_dicts([plt.default_plot_config, config['metrics'][metric_name]])
    plot_config_string = "plot_config = {}\n"
    for key in plot_config:
        if type(plot_config[key]) == str:
            plot_config_string = plot_config_string + f"plot_config[\"{key}\"] = \"{plot_config[key]}\"\n"
        else:
            plot_config_string = plot_config_string + f"plot_config[\"{key}\"] = {plot_config[key]}\n"
    plot_type = config['metrics'][metric_name]['type']

    # Generate string of all functions needed for plotting
    util_functions = [plt.calc_gmean,
                      plt.save_or_show_figure,
                      plt.generic_plot,
                      plt.config_value,
                      run_script,
                      get_model_benchmark_data]
    if plot_type == 'bar':
        util_functions.extend([plt.bar_plot])
    if plot_type == 'stacked_bar':
        util_functions.extend([plt.set_plot_xticks, plt.stacked_bar_plot])
    if plot_type == 'violin':
        util_functions.extend([plt.set_plot_xticks, plt.violin_plot])
    util_functions_string = ""
    for function in util_functions:
        util_functions_string = util_functions_string + inspect.getsource(function) + "\n"

    # Code to read data and plot
    if plot_type == 'bar':
        plotting_code = """data_read_function = partial(get_model_benchmark_data, data_dir=data_dir, script=get_metric, metric=metric_name)
with ThreadPoolExecutor() as e:
    data = list(e.map(data_read_function, itertools.product(models,benchmarks)))
    data = np.array(list(e.map(lambda x: float(x), data)))
    data = data.reshape(len(models), len(benchmarks))
plot_data = dict(zip(model_names,data.tolist()))
bar_plot(plot_data, benchmarks, plot_config, filename=output_file_name)"""
    elif plot_type == 'stacked_bar':
        plotting_code = """data_read_function = partial(get_model_benchmark_data, data_dir=data_dir, script=get_metric, metric=metric_name)
with ThreadPoolExecutor() as e:
    data = e.map(data_read_function, itertools.product(models,benchmarks))
    data = np.array(list(e.map (lambda y: [float(z) for z in y],
                                e.map(lambda x: x.rstrip().split(","), data))), dtype=list)
    data = data.reshape(len(models), len(benchmarks), -1)
    transposed_data = []
    for model_data in data:
        transposed_data.append(model_data.transpose().tolist())
plot_data = dict(zip(model_names,transposed_data))
stacked_bar_plot(plot_data, benchmarks, plot_config, filename=output_file_name)"""
    elif plot_type == 'violin':
        plotting_code = """data_read_function = partial(get_model_benchmark_data, data_dir=data_dir, script=get_metric, metric=metric_name)
with ThreadPoolExecutor() as e:
    data = e.map(data_read_function, itertools.product(models,benchmarks))
    data = np.array(list(e.map (lambda y: [float(z) for z in y],
                                e.map(lambda x: x.rstrip().split(" "), data))), dtype=list)
    data = data.reshape(len(models), len(benchmarks))
plot_data = dict(zip(model_names,data.tolist()))
violin_plot(plot_data, benchmarks, plot_config, filename=output_file_name)"""
    else:
        print(f"Generating plot scripts is currently not supported for {plot_type} plots")
        exit()

    if savefig:
        savefig = f"'{savefig}'"
    if 'model_names' not in config:
        config['model_names'] = {}

    with open(filename, "w") as outfile:
        outfile.write(template.render(
            models = models,
            benchmarks = benchmarks,
            metric_name = f"'{metric_name}'",
            data_dir = f"'{config['data_dir']}'",
            get_metric = f"'{config['hooks']['get_metric']}'",
            output_file_name = savefig,
            model_names = config['model_names'],
            plot_config = plot_config_string,
            util_functions = util_functions_string,
            plotting_code = plotting_code,
        ))

@cli.command()
@click.option("--benchmarks", "-b", help="Comma separated list of benchmarks to plot")
@click.option("--models", "-m", required=True, help="Comma separated list of models to plot")
@click.option("--metrics", "-M", required=True, help="Comma separated list of metrics to plot")
@click.option("--savefig", "-s", help="Filename to save the plot. If this option is not specified, the plot is displayed")
@click.option("--genplot", "-g", help="Generate a python plot script with the given file name")
@click.pass_context
def plot(ctx, benchmarks, models, metrics, savefig, genplot):
    config = ctx.obj['config']
    data_dir = config['data_dir']

    print("Benchmarks to plot: ",end='')
    if not benchmarks:
        benchmarks = "*"
    benchmarks = process_glob_patterns(config['benchmarks'], benchmarks)

    print("Models to plot: ",end='')
    models = process_glob_patterns(config['models'], models)
    model_names = []
    if 'model_names' in config:
        for model in models:
            if model in config['model_names']:
                model_names.append(config['model_names'][model])
            else:
                model_names.append(model)
    else:
        for model in models:
            model_names.append(model)

    metrics = metrics.split(",")
    metrics_to_calculate = []
    for metric in metrics:
        if metric not in config['metrics'] and metric not in config['metric_groups']:
            print(f"The Metric {metric} is not found in retconfig")
            continue
        elif metric in config['metric_groups']:
            # Valid Metric group
            for m in config['metric_groups'][metric]:
                metrics_to_calculate.append(m)
        else:
            # Valid metric
            metrics_to_calculate.append(metric)
    metrics = metrics_to_calculate

    print("Metrics to plot: ",end='')
    for metric in metrics:
        print(f"{metric} ", end='')
    print()

    if genplot:
        if len(metrics) != 1:
            print("Plot script generation only works with a single metric")
            exit()
        generate_plot_script(models, benchmarks, metrics[0], config, savefig, filename=genplot)
        exit()

    for metric in metrics:
        data_read_function = partial(get_model_benchmark_data, data_dir=config['data_dir'], script=config['hooks']['get_metric'], metric=metric)
        if 'default_plot_config' in config:
            plot_config = merge_dicts([plt.default_plot_config, config['default_plot_config'], config['metrics'][metric]])
        else:
            plot_config = merge_dicts([plt.default_plot_config, config['metrics'][metric]])
        if plot_config['type'] == 'bar':
            with ThreadPoolExecutor() as e:
                data = e.map(data_read_function, itertools.product(models,benchmarks))
                data = np.array(list(e.map(lambda x: float(x), data)))
                data = data.reshape(len(models), len(benchmarks))
            plot_data = dict(zip(model_names,data.tolist()))
            plt.bar_plot(plot_data, benchmarks, plot_config, filename=savefig)
        elif config['metrics'][metric]['type'] == 'cdf':
            title = metric
            if 'title' in config['metrics'][metric]:
                title = config['metrics'][metric]['title']
            for model in models:
                for benchmark in benchmarks:
                    run_dir = os.path.join(data_dir, model, benchmark)
                    comma_separated_values = run_hook(config, 'get_metric', [model, benchmark, run_dir, metric], capture_output=True)
                    data = list(map(int, comma_separated_values.split(",")))
                    val, cnts = np.unique(data, return_counts=True)
                    mplt.plot(val,np.cumsum(cnts))
                    mplt.title = f"{title} : {benchmark}"
                    mplt.show()
        elif plot_config['type'] == 'stacked_bar':
            with ThreadPoolExecutor() as e:
                data = e.map(data_read_function, itertools.product(models,benchmarks))
                data = np.array(list(e.map (lambda y: [float(z) for z in y],
                                            e.map(lambda x: x.rstrip().split(","), data))), dtype=list)
                data = data.reshape(len(models), len(benchmarks), -1)
                transposed_data = []
                for model_data in data:
                    transposed_data.append(model_data.transpose().tolist())
            plot_data = dict(zip(model_names,transposed_data))
            plt.stacked_bar_plot(plot_data, benchmarks, plot_config, filename=savefig)
        elif plot_config['type'] == 'stacked_bar_per_run':
            with ThreadPoolExecutor() as e:
                data = list(e.map(data_read_function, itertools.product(models,benchmarks)))

            index = 0
            original_title = ""
            if 'title' in plot_config:
                original_title = plot_config['title']
            with ThreadPoolExecutor() as e:
                for model in models:
                    for benchmark in benchmarks:
                        epoch_data = data[index].strip().split(":")
                        epoch_data = np.array(list(e.map (lambda y: [float(z) for z in y],
                                                          e.map(lambda x: x.rstrip().split(","), epoch_data))), dtype=list)
                        epoch_data = epoch_data.transpose().tolist()
                        plot_config['title'] = original_title.format(model=model,benchmark=benchmark)
                        plt.stacked_bar_plot({model:epoch_data}, ["" for _ in range(len(epoch_data[0]))], plot_config, filename=savefig)
                        index += 1
        elif config['metrics'][metric]['type'] == 'violin':
            with ThreadPoolExecutor() as e:
                data = e.map(data_read_function, itertools.product(models,benchmarks))
                data = np.array(list(e.map (lambda y: [float(z) for z in y],
                                            e.map(lambda x: x.rstrip().split(" "), data))), dtype=list)
                data = data.reshape(len(models), len(benchmarks))
            plot_data = dict(zip(model_names,data.tolist()))
            plt.violin_plot(plot_data, benchmarks, plot_config, filename=savefig)
        elif config['metrics'][metric]['type'] == 'lines_per_run':
            if 'title' in config['metrics'][metric]:
                title = config['metrics'][metric]['title']
            if 'line_labels' in config['metrics'][metric]:
                line_labels = config['metrics'][metric]['line_labels']
            else:
                line_labels = []
            for model in models:
                for benchmark in benchmarks:
                    run_dir = os.path.join(data_dir, model, benchmark)
                    line_datas = list(run_hook(config, 'get_metric', [model, benchmark, run_dir, metric], capture_output=True).strip().split(":"))

                    plot_data = []
                    for line in line_datas:
                        plot_data.append([float(x) for x in line.split(",")])

                    x_vals = list(range(1,len(plot_data[0])+1))
                    fig, ax = mplt.subplots()
                    ax.set_title(f"{benchmark} : {title}")
                    for i in range(0,len(line_datas)):
                        if i < len(line_labels):
                            ax.plot(x_vals, plot_data[i], label=line_labels[i])
                        else:
                            ax.plot(x_vals, plot_data[i])
                    ax.legend()
                    mplt.show()
        elif config['metrics'][metric]['type'] == 'script':
            for model in models:
                for benchmark in benchmarks:
                    run_dir = os.path.join(data_dir, model, benchmark)
                    list(run_hook(config, 'get_metric', [model, benchmark, run_dir, metric], capture_output=True))

cli()
