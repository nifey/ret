# Ret Tool

Tool to help in running experiments and saving logs

Ret automates running experiments on a set of models using a set of benchmarks.
For each project, we need to write scripts that are run at different hook points.

Ret can also retrieve metrics from logs and present them as a plot (from a set of standard plot types).
This will be useful to quickly visualize experiment results and compare select models on select metrics.
Ret can also generate a standard plot script that can then be customized.

## Installation
```python
pip install rettool
```

## Usage
- ret run -m model1,model2 -b benchmark1,benchmark2
- ret plot -M metric_to_plot -m model1,model2 -b benchmark1,benchmark2
 
## Project information specified in retconfig.yml
Ret finds project information from **retconfig.yml** file in the current directory.
The project information includes the list of models, benchmarks, metrics, run constraints, etc.
An example retconfig.yml file in given in [here](./example_project/retconfig.yml).

## Hooks

## Workflow
Models, Benchmarks, Metrics
