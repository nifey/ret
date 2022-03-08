import click
import yaml

@click.group()
def cli():
    # Read and parse retconfig.yml from current directory
    with open("retconfig.yml", 'r') as configfile:
        try:
            config = yaml.safe_load(configfile)
        except yaml.YAMLError as err:
            print(err)
            print("Error in parsing retconfig.yml")
            exit(0)

@cli.command()
@click.option("--benchmarks", "-b", help="Comma separated list of benchmarks to run")
@click.option("--models", "-m", required=True, help="Comma separated list of benchmarks to run")
def run(benchmarks, models):
    print("Models to run    : ",end='')
    for model in models.split(","):
        print(f"{model} ", end='')
    print()
    if benchmarks:
        print("Benchmarks to run: ",end='')
        for benchmark in benchmarks.split(","):
            print(f"{benchmark} ", end='')
        print()
    else:
        # FIXME Run all benchmarks if none specified
        pass

cli()
