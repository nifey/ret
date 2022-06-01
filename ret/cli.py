import click
import os
import subprocess
import yaml
import numpy as np
import matplotlib.pyplot as mplt
import plot as plt

from functools import partial
from concurrent.futures import ProcessPoolExecutor

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

def execute_run(benchmark, config, model_name, data_dir):
    # Create a folder for this run
    run_dir = os.path.join(os.getcwd(), data_dir, f"{model_name}_{benchmark}")
    os.mkdir(run_dir)
    # Run hooks
    arguments = [model_name, benchmark, run_dir]
    run_hook(config, 'pre_run', arguments)
    run_hook(config, 'run', arguments)
    run_hook(config, 'post_run', arguments)

@cli.command()
@click.option("--benchmarks", "-b", help="Comma separated list of benchmarks to run")
@click.option("--model-name", "-m", required=True, help="Name of current model")
@click.pass_context
def run(ctx, benchmarks, model_name):
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

    run_hook(config, 'pre_batch', [model_name, data_dir])

    run_function = partial(execute_run, config=config, model_name=model_name, data_dir=data_dir)
    with ProcessPoolExecutor() as executor:
        list(executor.map(run_function,
                          benchmarks))

    run_hook(config, 'post_batch', [model_name, data_dir])

@cli.command()
@click.option("--benchmarks", "-b", help="Comma separated list of benchmarks to plot")
@click.option("--models", "-m", required=True, help="Comma separated list of models to plot")
@click.option("--metrics", "-M", required=True, help="Comma separated list of metrics to plot")
@click.pass_context
def plot(ctx, benchmarks, models, metrics):
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
    print("Models to plot: ",end='')
    for model in models:
        print(f"{model} ", end='')
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
        # Collect data for metric
        if config['metrics'][metric]['type'] == 'bar':
            title = metric
            if 'title' in config['metrics'][metric]:
                title = config['metrics'][metric]['title']
            gmean = False
            if 'gmean' in config['metrics'][metric]:
                gmean = config['metrics'][metric]['gmean']
            plot_data = {}
            for model in models:
                plot_data[model] = []
                for benchmark in benchmarks:
                    plot_data[model].append(0)
                for benchmark in benchmarks:
                    run_dir = os.path.join(os.getcwd(), data_dir, f"{model}_{benchmark}")
                    data = run_hook(config, 'get_metric', [model, benchmark, run_dir, metric], capture_output=True)
                    plot_data[model][benchmarks.index(benchmark)] = float(data)
            plt.bar_plot(plot_data, benchmarks, title=title, gmean=gmean)
        elif config['metrics'][metric]['type'] == 'cdf':
            title = metric
            if 'title' in config['metrics'][metric]:
                title = config['metrics'][metric]['title']
            for model in models:
                for benchmark in benchmarks:
                    run_dir = os.path.join(os.getcwd(), data_dir, f"{model}_{benchmark}")
                    comma_separated_values = run_hook(config, 'get_metric', [model, benchmark, run_dir, metric], capture_output=True)
                    data = list(map(int, comma_separated_values.split(",")))
                    val, cnts = np.unique(data, return_counts=True)
                    mplt.plot(val,np.cumsum(cnts))
                    mplt.title = f"{title} : {benchmark}"
                    mplt.show()
        elif config['metrics'][metric]['type'] == 'stacked_bar':
            title = metric
            if 'title' in config['metrics'][metric]:
                title = config['metrics'][metric]['title']
            plot_bar_labels = config['metrics'][metric]['stack_labels']
            for model in models:
                plot_data = []
                for stack_bar in plot_bar_labels:
                    plot_data.append(np.zeros(len(benchmarks)))
                for benchmark in benchmarks:
                    run_dir = os.path.join(os.getcwd(), data_dir, f"{model}_{benchmark}")
                    comma_separated_values = run_hook(config, 'get_metric', [model, benchmark, run_dir, metric], capture_output=True)
                    for i, data_item in enumerate(list(map(float, comma_separated_values.split(",")))):
                        plot_data[i][benchmarks.index(benchmark)] = data_item

                bottom = np.zeros(len(benchmarks))
                fig, ax = mplt.subplots()
                ax.set_title(title)
                for i, stack_bar in enumerate(plot_bar_labels):
                    ax.bar(benchmarks, plot_data[i], label=stack_bar, bottom=bottom)
                    bottom += plot_data[i]
                ax.legend()
                mplt.show()
        elif config['metrics'][metric]['type'] == 'stacked_bar_per_run':
            title = metric
            if 'title' in config['metrics'][metric]:
                title = config['metrics'][metric]['title']
            plot_bar_labels = config['metrics'][metric]['stack_labels']
            for model in models:
                for benchmark in benchmarks:
                    run_dir = os.path.join(os.getcwd(), data_dir, f"{model}_{benchmark}")
                    epoch_data = list(run_hook(config, 'get_metric', [model, benchmark, run_dir, metric], capture_output=True).strip().split(":"))

                    plot_data = []
                    for i in range(len(plot_bar_labels)):
                        plot_data.append([])

                    for epoch in epoch_data:
                        for i, num in enumerate(epoch.split(",")):
                            plot_data[i].append(float(num))

                    bottom = np.zeros(len(plot_data[0]))
                    x_vals = list(range(1,len(plot_data[0])+1))
                    fig, ax = mplt.subplots()
                    ax.set_title(f"{benchmark} : {title}")
                    for i, stack_bar in enumerate(plot_bar_labels):
                        ax.bar(x_vals, plot_data[i], label=stack_bar, bottom=bottom)
                        bottom += plot_data[i]
                    ax.legend()
                    mplt.show()
        elif config['metrics'][metric]['type'] == 'violin':
            title = metric
            if 'title' in config['metrics'][metric]:
                title = config['metrics'][metric]['title']
            fig, ax = mplt.subplots()
            ax.set_title(f"{title}")
            ax.set_xticks(range(1,len(benchmarks)+1), benchmarks)
            for model in models:
                plot_data = []
                for benchmark in benchmarks:
                    plot_data.append([])
                for benchmark in benchmarks:
                    run_dir = os.path.join(os.getcwd(), data_dir, f"{model}_{benchmark}")
                    data = run_hook(config, 'get_metric', [model, benchmark, run_dir, metric], capture_output=True)
                    data = list(map(float, data.strip().split(" ")))
                    plot_data[benchmarks.index(benchmark)] = data
                ax.violinplot(plot_data, showmeans=True)
            mplt.show()
        elif config['metrics'][metric]['type'] == 'lines_per_run':
            if 'title' in config['metrics'][metric]:
                title = config['metrics'][metric]['title']
            line_labels = config['metrics'][metric]['line_labels']
            for model in models:
                for benchmark in benchmarks:
                    run_dir = os.path.join(os.getcwd(), data_dir, f"{model}_{benchmark}")
                    line_datas = list(run_hook(config, 'get_metric', [model, benchmark, run_dir, metric], capture_output=True).strip().split(":"))

                    plot_data = []
                    for line in line_datas:
                        plot_data.append([float(x) for x in line.split(",")])

                    x_vals = list(range(1,len(plot_data[0])+1))
                    fig, ax = mplt.subplots()
                    ax.set_title(f"{benchmark} : {title}")
                    for i, line_label in enumerate(line_labels):
                        ax.plot(x_vals, plot_data[i], label=line_label)
                    ax.legend()
                    mplt.show()
 

cli()
