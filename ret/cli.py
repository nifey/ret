import click
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

@cli.command()
@click.option("--benchmarks", "-b", help="Comma separated list of benchmarks to run")
@click.option("--models", "-m", required=True, help="Comma separated list of benchmarks to run")
@click.pass_context
def run(ctx, benchmarks, models):
    print("Models to run    : ",end='')
    for model in models.split(","):
        print(f"{model} ", end='')
    print()

    if benchmarks:
        benchmarks = benchmarks.split(",")
    else:
        benchmarks = ctx.obj['config']['benchmarks']
    print("Benchmarks to run: ",end='')
    for benchmark in benchmarks:
        print(f"{benchmark} ", end='')
    print()

cli()
