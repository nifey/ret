import functools
import math
import matplotlib.pyplot as plt
import numpy as np

def calc_gmean(numbers):
    sum_of_logs = functools.reduce(lambda x, y: x+y,
                              map(math.log, numbers))
    return pow(math.e, sum_of_logs/len(numbers))

def bar_plot(data, xticks, title=None, filename=None, gmean=False, ylabel=""):
    """Create a Bar plot

    Takes as argument a dict from (model => [model_value for each benchmark])
    """
    if gmean:
        for model in data.keys():
            data[model].append(calc_gmean(data[model]))

    total_width = 0.8
    bar_width = total_width/len(data)
    for i, model in enumerate(data.keys()):
        current_plot_data = data[model]
        start_offset = (total_width/2) - (2*i+1)*bar_width/2
        plt.bar(np.arange(len(current_plot_data)) - start_offset, current_plot_data, width=bar_width, label=model)
    plt.title(title or "Bar chart")
    plt.ylabel(ylabel)
    xticks = xticks[:]
    if gmean:
        xticks.append("gmean")
    plt.xticks(np.arange(len(xticks)), xticks)
    plt.legend()
    #plt.savefig(filename or f"plots/{title}.jpg")
    plt.show()

def stacked_bar_plot():
    pass

def violin_plot(data, title=None, xticks=None, filename=None):
    plt.violinplot(data, showmeans=True)
    plt.title(title or "Violin Plot")
    if xticks:
        plt.xticks(np.arange(len(xticks))+1, xticks)
    plt.legend()
    plt.savefig(f"plots/{title}.jpg")
    plt.show()
