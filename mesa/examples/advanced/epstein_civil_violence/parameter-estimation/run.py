import pandas as pd
import json
import numpy as np
from SALib.analyze import sobol
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.legend_handler import HandlerTuple

def load_base(series):
    return np.array(json.loads(series))

def load_x(x=-1):
    def load(series):
        return json.loads(series)[x]
    return load

def merge(files, load):
    files = ["0002_data_0000.csv", "0002_data_0001.csv", "0002_data_0002.csv"]
    all_data_list = []
    for file in files:
        filename = f"./processed/{file}"
        data = pd.read_csv(filename)
        all_data_list.append(data)
    all_data = pd.concat(all_data_list)
    all_data["police"] = all_data["police"].apply(load)
    all_data["citizen"] = all_data["citizen"].apply(load)
    all_data = all_data.drop("Unnamed: 0", axis=1)
    all_data = all_data.drop("Unnamed: 0.1", axis=1)
    return all_data

def plot_index(s, params, i, title=''):
    """
    Creates a plot for Sobol sensitivity analysis that shows the contributions
    of each parameter to the global sensitivity.

    Args:
        s (dict): dictionary {'S#': dict, 'S#_conf': dict} of dicts that hold
            the values for a set of parameters
        params (list): the parameters taken from s
        i (str): string that indicates what order the sensitivity is.
        title (str): title for the plot
    """

    if i == '2':
        p = len(params)
        params = list(combinations(params, 2))
        indices = s['S' + i].reshape((p ** 2))
        indices = indices[~np.isnan(indices)]
        errors = s['S' + i + '_conf'].reshape((p ** 2))
        errors = errors[~np.isnan(errors)]
    else:
        indices = s['S' + i]
        errors = s['S' + i + '_conf']
        plt.figure()

    l = len(indices)

    plt.title(title)
    plt.ylim([-0.2, len(indices) - 1 + 0.2])
    plt.yticks(range(l), params)
    plt.errorbar(indices, range(l), xerr=errors, linestyle='None', marker='o')
    plt.axvline(0, c='k')
    plt.tight_layout()

def SA():
    problem = {
        'num_vars': 4,
        'names': ['legitimacy', 'active_threshold', 'reversion_rate', 'prob_quiet'],
        'bounds': [[0.1, 1.0], [0.0, 0.9], [0.0, 1.0], [0.0, 0.5]]
    }
    files = ["0002_data_0000.csv", "0002_data_0001.csv", "0002_data_0002.csv"]
    data = merge(files, load_x(20))
    Si_police = sobol.analyze(problem, data["police"].values, print_to_console=True)
    Si_citizen = sobol.analyze(problem, data["citizen"].values, print_to_console=True)
    # First order
    plot_index(Si_police, problem['names'], '1', 'First order sensitivity')
    plt.savefig(f"./img/first_order_police.pdf")
    plt.clf()

    # Second order
    plot_index(Si_police, problem['names'], '2', 'Second order sensitivity')
    plt.savefig(f"./img/second_order_police.pdf")
    plt.clf()

    # Total order
    plot_index(Si_police, problem['names'], 'T', 'Total order sensitivity')
    plt.savefig(f"./img/total_order_police.pdf")
    plt.clf()

    # First order
    plot_index(Si_citizen, problem['names'], '1', 'First order sensitivity')
    plt.savefig(f"./img/first_order_citizen.pdf")
    plt.clf()

    # Second order
    plot_index(Si_citizen, problem['names'], '2', 'Second order sensitivity')
    plt.savefig(f"./img/second_order_citizen.pdf")
    plt.clf()

    # Total order
    plot_index(Si_citizen, problem['names'], 'T', 'Total order sensitivity')
    plt.savefig(f"./img/total_order_citizen.pdf")
    plt.clf()

def get_10_90_percentiles(data):
    k10 = np.zeros(len(data[0]))
    k90 = np.zeros(len(data[0]))
    for i in range(len(data[0])):
        data[:, i] = np.sort(data[:, i])
        k10[i] = data[:, i][9]
        k90[i] = data[:, i][89]
    return k10, k90

def plot_agents(data, title, filename, column_source, columns, k10, k90, radii):
    plots = []
    for name, group in data:
        mean = np.mean(group[column_source].to_list(), axis=0)
        p = plt.errorbar(
            radii,
            mean,
            #yerr=police_std,
            #label=f"Police L-function {name}",
            fmt="-o",
            capsize=5,
            color="red",
            zorder=0
        )
        plots.append(p)
    fill = plt.fill_between(
        radii,
        k10,
        k90,
        color="purple",
        alpha=0.6,
        label="Police 10-90% range",
        zorder=5
    )
    plt.xlabel("Radius")
    plt.ylabel("Ripley's L-function")
    plt.title(title)
    plt.legend([tuple(plots), fill], columns, handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.grid()
    plt.savefig(f"./img/{filename}.pdf")
    plt.clf()

def plot():
    radii = np.linspace(0.1, 40 / 2, 50)
    files = ["0002_data_0000.csv", "0002_data_0001.csv", "0002_data_0002.csv"]
    data = merge(files, load_base)
    d = data.groupby(["legitimacy","active_threshold","reversion_rate","prob_quiet"])
    base_citizen = []
    base_police = []
    with open("./processed/test_citizen.json", "r") as f:
        base_citizen = json.load(f)
    with open("./processed/test_police.json", "r") as f:
        base_police = json.load(f)
    k10_citizen, k90_citizen = get_10_90_percentiles(np.array(base_citizen))
    k10_police, k90_police = get_10_90_percentiles(np.array(base_police))

    plot_agents(d, "Ripley's L-function for Citizen", "ripley_l_function_citizen", "citizen", ["Citizen", "Baseline"], k10_citizen, k90_citizen, radii)
    plot_agents(d, "Ripley's L-function for Police", "ripley_l_function_police", "police", ["Police", "Baseline"], k10_police, k90_police, radii)

def transpose(list_of_lists):
    return [list(x) for x in zip(*list_of_lists)]

SA()
plot()