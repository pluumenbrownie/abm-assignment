import pandas as pd
import json
import numpy as np
from SALib.analyze import sobol
from SALib.util.results import ResultDict
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.legend_handler import HandlerTuple

files = ["0002_data_0000_0.csv", "0002_data_0000_1.csv", "0002_data_0001_0.csv", "0002_data_0001_1.csv", "0002_data_0002.csv"]

def load_base(series):
    """
    Load a series of data from a JSON string into a numpy array.
    
    Parameters
    ----------
    series : str
        A JSON string representing a list of agent coordinates in the form of
        `[[x1, y1], [x2, y2], ...]`.
    
    Returns
    -------
    np.ndarray
        A numpy array of the agent coordinates.
    """
    return np.array(json.loads(series))

def load_x(x=-1):
    """
    Load a specific index from a JSON string in a pandas series.
    
    Parameters
    ----------
    x : int, optional
        The index of the value to extract from the JSON string. Defaults to -1,
        which retrieves the last element.
    
    Returns
    -------
    function
        A function that takes a pandas series and returns the value at index x
        from the JSON string.
    """
    def load(series):
        return json.loads(series)[x]
    return load

def merge(files, load):
    """
    Merge multiple CSV files into a single DataFrame, applying a loading function
    to specific columns.
    Parameters
    ----------
        files : list
            List of CSV filenames to merge.
        load : function
            Function to apply to the 'police' and 'citizen' columns.
    Returns
    -------
        pd.DataFrame : dataframe
            A DataFrame containing the merged data with the specified columns processed.
    """
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

def plot_all_indices(analyses: dict[str, ResultDict], params, title=""):
    """
    Creates a plot for Sobol sensitivity analysis that shows the contributions
    of each parameter to the global sensitivity.

    Args:
        analyses (dict[str, ResultDict]): dictionary mapping the name of the analysis to
            dictionaries {'S#': dict, 'S#_conf': dict} of dicts that hold
            the values for a set of parameters
        params (list): the names of the parameters in the analyses
        title (str): title for the plot
    """
    single_names = params
    second_names = [f"{x}+\n{y}" for x, y in combinations(params, 2)]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout="compressed")

    for nr, analysis in enumerate(analyses.items()):
        name, s = analysis
        for i in ["1", "2", "T"]:
            # first order
            if i == "1":
                indices = s["S" + i]
                errors = s["S" + i + "_conf"]
                axes[0].errorbar(
                    indices,
                    np.arange(len(indices)) + nr * 0.1,
                    xerr=errors,
                    linestyle="None",
                    marker="o",
                    label=name,
                )
            # second order
            elif i == "2":
                flattened = s["S" + i].flatten()
                indices = flattened[~np.isnan(flattened)]

                flattened = s["S" + i + "_conf"].flatten()
                errors = flattened[~np.isnan(flattened)]
                axes[1].errorbar(
                    indices,
                    np.arange(len(indices)) + nr * 0.1,
                    xerr=errors,
                    linestyle="None",
                    marker="o",
                    label=name,
                )
            # total order
            else:
                indices = s["S" + i]
                errors = s["S" + i + "_conf"]
                axes[2].errorbar(
                    indices,
                    np.arange(len(indices)) + nr * 0.1,
                    xerr=errors,
                    linestyle="None",
                    marker="o",
                    label=name,
                )

    axes[0].set_yticks(range(len(single_names)), single_names, size=9)
    axes[1].set_yticks(range(len(second_names)), second_names, size=9)
    axes[2].set_yticks(range(len(single_names)), single_names, size=9)

    axes[0].axvline(0, c="k")
    axes[1].axvline(0, c="k")
    axes[2].axvline(0, c="k")

    axes[0].set_title("First order sensitivity")
    axes[1].set_title("Second order sensitivity")
    axes[2].set_title("Total order sensitivity")

    axes[2].legend()
    fig.suptitle(title)

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
    """ Perform a global sensitivity analysis using Sobol's method on the Epstein Civil Violence model.
    This function sets up the problem parameters, loads the data from CSV files,
    and computes the sensitivity indices for both police and citizen agents.
    It then generates plots for first-order, second-order, and total order sensitivity indices."""
    problem = {
        'num_vars': 4,
        'names': ['legitimacy', 'active_threshold', 'reversion_rate', 'prob_quiet'],
        'bounds': [[0.1, 1.0], [0.0, 0.9], [0.0, 1.0], [0.0, 0.5]]
    }
    data = merge(files, load_x(20))
    Si_police = sobol.analyze(problem, data["police"].values, print_to_console=True)
    Si_citizen = sobol.analyze(problem, data["citizen"].values, print_to_console=True)

    plot_all_indices(
        {"Police": Si_police, "Citizen": Si_citizen},
        problem["names"],
        "Global sensitivity analysis",
    )
    plt.savefig(f"./img/global_sensitivity_analysis.pdf")
    plt.clf()
    # # First order
    # plot_index(Si_police, problem['names'], '1', 'First order sensitivity')
    # plt.savefig(f"./img/first_order_police.pdf")
    # plt.clf()

    # # Second order
    # plot_index(Si_police, problem['names'], '2', 'Second order sensitivity')
    # plt.savefig(f"./img/second_order_police.pdf")
    # plt.clf()

    # # Total order
    # plot_index(Si_police, problem['names'], 'T', 'Total order sensitivity')
    # plt.savefig(f"./img/total_order_police.pdf")
    # plt.clf()

    # # First order
    # plot_index(Si_citizen, problem['names'], '1', 'First order sensitivity')
    # plt.savefig(f"./img/first_order_citizen.pdf")
    # plt.clf()

    # # Second order
    # plot_index(Si_citizen, problem['names'], '2', 'Second order sensitivity')
    # plt.savefig(f"./img/second_order_citizen.pdf")
    # plt.clf()

    # # Total order
    # plot_index(Si_citizen, problem['names'], 'T', 'Total order sensitivity')
    # plt.savefig(f"./img/total_order_citizen.pdf")
    # plt.clf()

def get_10_90_percentiles(data):
    """
    Calculate the 10th and 90th percentiles for each column in the data.
    
    Parameters
    ----------
    data : np.ndarray
        A 2D numpy array where each column represents a different set of data points.
    Returns
    -------
    k10 : np.ndarray
        A 1D numpy array containing the 10th percentiles for each column.
    k90 : np.ndarray
        A 1D numpy array containing the 90th percentiles for each column.
    """
    k10 = np.zeros(len(data[0]))
    k90 = np.zeros(len(data[0]))
    for i in range(len(data[0])):
        data[:, i] = np.sort(data[:, i])
        k10[i] = data[:, i][9]
        k90[i] = data[:, i][89]
    return k10, k90

def plot_agents(data, title, filename, column_source, columns, k10, k90, radii):
    """
    Plot Ripley's L-function for a given set of agents.
    
    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the agent data grouped by parameters.
    title : str
        The title of the plot.
    filename : str
        The filename to save the plot as.
    column_source : str
        The column name in the DataFrame that contains the agent coordinates.
    columns : list
        A list of column names to be used in the legend.
    k10 : np.ndarray
        A 1D numpy array containing the 10th percentiles for each radius.
    k90 : np.ndarray
        A 1D numpy array containing the 90th percentiles for each radius.
    radii : np.ndarray
        A 1D numpy array of radii at which the L-function is computed.
    """
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
    """
    Plot Ripley's L-function for citizen and police agents based on the data
    collected from multiple CSV files. This function merges the data, computes
    the 10th and 90th percentiles, and generates plots for both citizen and
    police agents, saving them as PDF files.
    """
    radii = np.linspace(0.1, 40 / 2, 50)
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
    """
    Transpose a list of lists.
    
    Parameters
    ----------
    list_of_lists : list of lists
        A list where each element is a list of the same length.
    
    Returns
    -------
    list of lists
        A transposed version of the input list, where each sublist contains
        elements from the same index across all sublists.
    """
    return [list(x) for x in zip(*list_of_lists)]

SA()
plot()