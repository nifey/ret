import functools
import math
import matplotlib.pyplot as plt
import numpy as np

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

hatches = ['//', '\\\\', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**', '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

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

def config_value(plot_config, name, default_value):
    if name in plot_config:
        return plot_config[name]
    else:
        return default_value

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

    # Bar plot config values
    bar_width = config_value(plot_config, 'bar_width', 1)
    intra_bar_gap = config_value(plot_config, 'intra_bar_gap', 0.1)
    inter_bar_gap = config_value(plot_config, 'inter_bar_gap', 0.5)
    annotate_outliers = config_value(plot_config, 'annotate_outliers', False)
    annotate_all = config_value(plot_config, 'annotate_all', False)
    annotation_gap = config_value(plot_config, 'annotation_gap', 0.01)
    xticks_rotation = config_value(plot_config, 'xticks_rotation', 0)
    xticks_ha = config_value(plot_config, 'xticks_horizontal_alignment', 'center')
    show_gmean = config_value(plot_config, 'gmean', False)
    show_hatch = config_value(plot_config, 'show_hatch', False)

    if show_gmean:
        xticks.append("gmean")
        for model in data.keys():
            gmean_value = calc_gmean(data[model])
            data[model].append(gmean_value)
            print(f"Gmean for {model}: {round(gmean_value,5)}")

    start_x = np.arange(len(xticks)) * (len(data) * bar_width + (len(data) - 1) * intra_bar_gap + inter_bar_gap)
    xticks_positions = start_x + (len(data) * bar_width + (len(data) - 1) * intra_bar_gap) / 2.0 - (bar_width/2.0)
    ax.set_xticks(xticks_positions, xticks, rotation=xticks_rotation, ha=xticks_ha)

    for i, model in enumerate(data.keys()):
        current_plot_data = data[model]
        x_values = start_x + i*(bar_width+intra_bar_gap)
        chosen_hatch = None
        if show_hatch:
            chosen_hatch = hatches[i]
        ax.bar(x_values, current_plot_data, width=bar_width, label=model, hatch=chosen_hatch, alpha=.99)
        max_val = ax.get_ylim()[1]
        for item_i, value in enumerate(current_plot_data):
            label = str.format("{:.2f}",round(value,2))
            if value > max_val:
                if annotate_all or annotate_outliers:
                    ax.annotate(label,
                                xy=(x_values[item_i], max_val + annotation_gap),
                                annotation_clip=False,
                                ha='center', va='bottom',rotation=90,size=15).draggable()
            elif annotate_all:
                ax.annotate(label,
                            xy=(x_values[item_i], value + annotation_gap),
                            annotation_clip=False,
                            ha='center', va='bottom',rotation=90,size=15).draggable()

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
    selected_hatches = {}
    bar_width = 0.8
    plot_bar_labels = plot_config['stack_labels']
    per_model_width = bar_width / len(models)
    initial_x_vals = np.array(range(0,len(xticks))) - 0.5 * bar_width + 0.5 * per_model_width
    for j, model in enumerate(models):
        bottom = np.zeros(len(xticks))
        x_vals = initial_x_vals + j * per_model_width
        for i, stack_bar in enumerate(plot_bar_labels):
            if stack_bar not in colors:
                selected_hatches[stack_bar] = None
                if plot_config['show_hatch']:
                    selected_hatches[stack_bar] = hatches[i]
                bar = ax.bar(x_vals, data[model][i], label=stack_bar, hatch=selected_hatches[stack_bar], bottom=bottom, width=per_model_width, alpha=.99)
                colors[stack_bar] = bar.patches[0].get_facecolor()
            else:
                ax.bar(x_vals, data[model][i], color=colors[stack_bar], hatch=selected_hatches[stack_bar], bottom=bottom, width=per_model_width, alpha=.99)
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
