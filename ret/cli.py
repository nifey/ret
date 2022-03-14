import click
import os
import subprocess
import yaml

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

def run_hook(config, hook_name, arguments):
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
    returncode = subprocess.run(command).returncode
    if returncode != 0:
        print(f"{hook_name} hook failed with error code {returncode}")
        exit(0)

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
    for benchmark in benchmarks:
        # Create a folder for this run
        run_dir = os.path.join(os.getcwd(), data_dir, f"{model_name}_{benchmark}")
        os.mkdir(run_dir)
        # Write general information to run_info file
        # Run hooks
        arguments = [model_name, benchmark, run_dir]
        run_hook(config, 'pre_run', arguments)
        run_hook(config, 'run', arguments)
        run_hook(config, 'post_run', arguments)
    run_hook(config, 'post_batch', [model_name, data_dir])

cli()
