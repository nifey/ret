import click
import os
import subprocess
import yaml
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
    print("Metrics to plot: ",end='')
    for metric in metrics:
        print(f"{metric} ", end='')
    print()

    for metric in metrics:
        # Collect data for metric
        plot_data = {}
        for model in models:
            plot_data[model] = []
            for benchmark in benchmarks:
                plot_data[model].append(0)
            for benchmark in benchmarks:
                run_dir = os.path.join(os.getcwd(), data_dir, f"{model}_{benchmark}")
                data = run_hook(config, 'get_metric', [model, benchmark, run_dir, metric], capture_output=True)
                plot_data[model][benchmarks.index(benchmark)] = float(data)
        plt.bar_plot(plot_data, benchmarks, title=metric, gmean=True)
cli()
