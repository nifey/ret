import functools
import math
import matplotlib.pyplot as plt
import numpy as np

def calc_gmean(numbers):
    sum_of_logs = functools.reduce(lambda x, y: x+y,
                              map(math.log, numbers))
    return pow(math.e, sum_of_logs/len(numbers))

def show_or_save_figure(mplt, filename):
    # If no filename was specified display the plot
    # else save the plot with the given filename
    if filename:
        aspect_ratio = .2
        x_left, x_right = mplt.xlim()
        y_low, y_high = mplt.ylim()
        mplt.gca().set_aspect(abs((x_right-x_left)/(y_low-y_high))*aspect_ratio)

        mplt.tight_layout()
        mplt.savefig(filename, bbox_inches='tight', pad_inches = 0.1)
    else:
        mplt.show()

def bar_plot(data, xticks, title=None, filename=None, gmean=False, ylabel="", ylim=None):
    """Create a Bar plot

    Takes as argument a dict from (model => [model_value for each benchmark])
    """
    if gmean:
        for model in data.keys():
            data[model].append(calc_gmean(data[model]))
    if ylim:
        plt.ylim(ylim)

    total_width = 0.8
    bar_width = total_width/len(data)
    for i, model in enumerate(data.keys()):
        current_plot_data = data[model]
        start_offset = (total_width/2) - (2*i+1)*bar_width/2
        plt.bar(np.arange(len(current_plot_data)) - start_offset, current_plot_data, width=bar_width, label=model)
        if ylim:
            for item_i, value in enumerate(current_plot_data):
                max_val = ylim[1]
                if value > max_val:
                    plt.annotate(str(round(value,2)),
                                 xy=(item_i - total_width/2 + (2*i+1)*(bar_width/2) - start_offset, max_val - (max_val*0.02)),
                                 ha='center').draggable()
    plt.title(title or "Bar chart")
    plt.ylabel(ylabel)
    xticks = xticks[:]
    if gmean:
        xticks.append("gmean")
    plt.xticks(np.arange(len(xticks)), xticks)
    plt.legend()
    show_or_save_figure(plt, filename)

def stacked_bar_plot():
    pass

def violin_plot(data, title=None, xticks=None, filename=None):
    plt.violinplot(data, showmeans=True)
    plt.title(title or "Violin Plot")
    if xticks:
        plt.xticks(np.arange(len(xticks))+1, xticks)
    plt.legend()
    show_or_save_figure(plt, filename)
