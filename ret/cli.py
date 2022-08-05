import click
import os
import subprocess
import yaml
import itertools
import numpy as np
import matplotlib.pyplot as mplt
import plot as plt

from functools import partial
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor

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

def run_hook(config, hook_name, arguments, capture_output=False):
    # Check if hook exists in config and in filesystem
    if hook_name not in config['hooks']:
        return
    script = config['hooks'][hook_name]
    if not os.path.exists(script):
        print(f"Script {script} does not exist")
        exit(0)
    # Run hook and retrieve return value
    command = [script]
    command.extend(arguments)
    completed_process = subprocess.run(command, capture_output=capture_output)
    returncode = completed_process.returncode
    if returncode != 0:
        print(f"{hook_name} hook failed with error code {returncode}")
        exit(0)
    if capture_output:
        return str(completed_process.stdout, 'UTF-8')

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

@cli.command()
@click.option("--benchmarks", "-b", help="Comma separated list of benchmarks to run")
@click.option("--models", "-m", required=True, help="Comma separated list of models to run")
@click.option("-j", help="Maximum number of runs to execute in parallel")
@click.pass_context
def run(ctx, benchmarks, models, j):
    config = ctx.obj['config']
    data_dir = config['data_dir']

    if benchmarks:
        benchmarks = benchmarks.split(",")
    else:
        benchmarks = config['benchmarks']
    print("Benchmarks to run: ",end='')
    for benchmark in benchmarks:
        print(f"{benchmark} ", end='')
    print()

    model_names = models
    models = models.split(",")
    print("Models to run: ",end='')
    for model in models:
        print(f"{model} ", end='')
    print()

    run_hook(config, 'pre_batch', [model_names, data_dir])

    # Create a ProcessPoolExecutor to run in parallel
    if config['run_contraint'] != 'serial':
        if j:
            j = int(j)
            executor = ProcessPoolExecutor(max_workers=j)
        else:
            executor = ProcessPoolExecutor()

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

def get_model_benchmark_data(mb_tuple, config, metric):
    model, benchmark = mb_tuple
    data_dir = config['data_dir']
    run_dir = os.path.join(data_dir, model, benchmark)
    data = run_hook(config, 'get_metric', [model, benchmark, run_dir, metric], capture_output=True)
    return data

@cli.command()
@click.option("--benchmarks", "-b", help="Comma separated list of benchmarks to plot")
@click.option("--models", "-m", required=True, help="Comma separated list of models to plot")
@click.option("--metrics", "-M", required=True, help="Comma separated list of metrics to plot")
@click.option("--savefig", "-s", help="Filename to save the plot. If this option is not specified, the plot is displayed")
@click.pass_context
def plot(ctx, benchmarks, models, metrics, savefig):
    config = ctx.obj['config']
    data_dir = config['data_dir']

    if benchmarks:
        benchmarks = benchmarks.split(",")
    else:
        benchmarks = config['benchmarks']
    print("Benchmarks to plot: ",end='')
    for benchmark in benchmarks:
        print(f"{benchmark} ", end='')
    print()

    models = models.split(",")
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

    print("Models to plot: ",end='')
    for model_name in model_names:
        print (f"{model_name} ",end='')
    print()

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

    for metric in metrics:
        data_read_function = partial(get_model_benchmark_data, config=config, metric=metric)
        if 'default_plot_config' in config:
            plot_config = plt.default_plot_config | config['default_plot_config'] | config['metrics'][metric]
        else:
            plot_config = plt.default_plot_config | config['metrics'][metric]
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
