# All imports
import os
import math
import yaml
import functools
import itertools
import subprocess
import numpy as np
import matplotlib.pyplot as plt

from functools import partial
from concurrent.futures import ThreadPoolExecutor

# Configs
models = {{ models }}
benchmarks = {{ benchmarks }}
metric_name = {{ metric_name }}
data_dir = {{ data_dir }}
get_metric = {{ get_metric }}
output_file_name = {{ output_file_name }}
model_names = {{ model_names }}

renamed_model_names = []
for model in models:
    if model in model_names:
        renamed_model_names.append(model_names[model])
    else:
        renamed_model_names.append(model)
model_names = renamed_model_names

# Plot config
{{ plot_config }}

# General matplotlib config
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14
plt.rc('axes', axisbelow=True)
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# Util functions
{{ util_functions }}

# Read data and plot
{{ plotting_code }}
