import functools
import math
import matplotlib.pyplot as plt
import numpy as np

# Default plot configs
default_plot_config = {
    "show_legend": True,
    "ygrid": True,

    # Bar plot
    "gmean": False,
}

def calc_gmean(numbers):
    sum_of_logs = functools.reduce(lambda x, y: x+y,
                              map(math.log, numbers))
    return pow(math.e, sum_of_logs/len(numbers))

def save_or_show_figure(plot_config, filename):
    """Save or show plot

    :param plot_config: Plot configuration
    :type plot_config: dict

    :param filename: File name to save the plot
    :type filename: str
    """
    # Legend
    if ('show_legend' in plot_config) and plot_config['show_legend']:
        plt.legend()

    if filename:
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', pad_inches = 0.1, dpi=300)
    else:
        plt.show()

def generic_plot(plot_config):
    """Create a plot and do generic configuration

    :param plot_config: Plot configuration
    :type plot_config: dict
    """
    # Figure size
    if 'figure_size' in plot_config:
        fig, ax = plt.subplots(figsize=(plot_config['figure_size']['width'], plot_config['figure_size']['height']))
    else:
        fig, ax = plt.subplots()

    # Title and Axes
    if 'title' in plot_config:
        ax.set_title(plot_config['title'])
    if 'xlabel' in plot_config:
        ax.set_xlabel(plot_config['xlabel'])
    if 'ylabel' in plot_config:
        ax.set_ylabel(plot_config['ylabel'])
    if ('ygrid' in plot_config) and plot_config['ygrid']:
        ax.grid(axis='y', which='major')
    if 'yscale' in plot_config:
        ax.set_yscale(plot_config['yscale'])

    # Y limits
    ylim = None
    if 'ylim' in plot_config:
        if 'min' in plot_config['ylim']:
            ax.set_ylim(bottom=float(plot_config['ylim']['min']))
        if 'max' in plot_config['ylim']:
            ax.set_ylim(top=float(plot_config['ylim']['max']))

    return fig, ax

def set_plot_xticks(ax, xticks, plot_config, data):
    """Configure and add xticks

    :param ax: Matplotlib axis
    :type ax: matplotlib Axis

    :param xticks: List of benchmarks
    :type xticks: list

    :param plot_config: Plot configuration
    :type plot_config: dict

    :param data: Dictionary with data to plot (needed to add gmean)
    :type data: dict
    """
    # Xticks
    xticks = xticks[:]
    if ('gmean' in plot_config) and (plot_config['gmean']):
        for model in data.keys():
            data[model].append(calc_gmean(data[model]))
        xticks.append("gmean")
    xticks_rotation = 0
    if 'xticks_rotation' in plot_config:
        xticks_rotation = plot_config['xticks_rotation']
    xticks_ha = 'center'
    if 'xticks_horizontal_alignment' in plot_config:
        xticks_ha = plot_config['xticks_horizontal_alignment']
    ax.set_xticks(np.arange(len(xticks)), xticks, rotation=xticks_rotation, ha=xticks_ha)

def bar_plot(data, xticks, plot_config, filename=None):
    """Create a Bar plot

    :param data: Dictionary with data to plot (model => [model_value for each benchmark])
    :type data: dict

    :param xticks: List of benchmarks
    :type xticks: list

    :param plot_config: Plot configuration
    :type plot_config: dict

    :param filename: File name to save the plot
    :type filename: str
    """
    fig, ax = generic_plot(plot_config)
    set_plot_xticks(ax, xticks, plot_config, data)

    annotate_outliers = False
    if 'annotate_outliers' in plot_config:
        annotate_outliers = plot_config['annotate_outliers']

    total_width = 0.8
    bar_width = total_width/len(data)

    for i, model in enumerate(data.keys()):
        current_plot_data = data[model]
        start_offset = (total_width/2) - (2*i+1)*bar_width/2
        ax.bar(np.arange(len(current_plot_data)) - start_offset, current_plot_data, width=bar_width, label=model)
        if annotate_outliers:
            for item_i, value in enumerate(current_plot_data):
                max_val = ax.get_ylim()[1]
                if value > max_val:
                    # FIXME accurate position calculation
                    ax.annotate(str(round(value,2)),
                                 xy=(item_i - total_width/2 + (2*i+1)*(bar_width/2) - start_offset, max_val - (max_val*0.02)),
                                 ha='center').draggable()

    save_or_show_figure(plot_config, filename)

def stacked_bar_plot(data, xticks, plot_config, filename=None):
    """Create a stacked bar plot

    :param data: Dictionary with data to plot (model => [[model_value for each benchmark] for each stack]))
    :type data: dict

    :param xticks: List of benchmarks
    :type xticks: list

    :param plot_config: Plot configuration
    :type plot_config: dict

    :param filename: File name to save the plot
    :type filename: str
    """
    fig, ax = generic_plot(plot_config)
    set_plot_xticks(ax, xticks, plot_config, data)

    models = data.keys()
    colors = {}
    bar_width = 0.8
    plot_bar_labels = plot_config['stack_labels']
    per_model_width = bar_width / len(models)
    initial_x_vals = np.array(range(0,len(xticks))) - 0.5 * bar_width + 0.5 * per_model_width
    for j, model in enumerate(models):
        bottom = np.zeros(len(xticks))
        x_vals = initial_x_vals + j * per_model_width
        for i, stack_bar in enumerate(plot_bar_labels):
            if stack_bar not in colors:
                bar = ax.bar(x_vals, data[model][i], label=stack_bar, bottom=bottom, width=per_model_width)
                colors[stack_bar] = bar.patches[0].get_facecolor()
            else:
                ax.bar(x_vals, data[model][i], color=colors[stack_bar], bottom=bottom, width=per_model_width)
            bottom += data[model][i]

    save_or_show_figure(plot_config, filename)

def violin_plot(data, xticks, plot_config, filename=None):
    """Create a violin plot

    :param data: Dictionary with data to plot (model => [list of values for each benchmark])
    :type data: dict

    :param xticks: List of benchmarks
    :type xticks: list

    :param plot_config: Plot configuration
    :type plot_config: dict

    :param filename: File name to save the plot
    :type filename: str
    """
    fig, ax = generic_plot(plot_config)
    set_plot_xticks(ax, xticks, plot_config, data)

    showmeans = False
    if 'show_means' in plot_config:
        showmeans = plot_config['show_means']

    for model in data.keys():
        model_data = data[model]
        plt.violinplot(model_data, positions=(list(range(len(xticks)))), showmeans=showmeans)

    save_or_show_figure(plot_config, filename)
