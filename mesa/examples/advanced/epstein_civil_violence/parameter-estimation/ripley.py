import pandas as pd
import json
import numpy as np
from scipy.spatial import distance_matrix
import os
import math
import sys
sys.path.append(os.path.abspath("../../../../.."))
from mesa.examples.advanced.epstein_civil_violence.model import EpsteinCivilViolence

def ripley_l_function(points, radii, area=None):
    """
    Compute Ripley's K-function for a set of agents.

    Parameters
    ----------
    points : array-like, shape (N, 2)
        The coordinates of the agents in the form of a 2D array.
    radii : array-like, shape (M,)
        The radii at which to compute the K-function.
    area : float, optional
        The area of the space in which the points are located.
        If not provided, it will be calculated based on the min and max coordinates of the points
    
    Returns
    L_r : array, shape (M,)
        The values of the L-function at the specified radii.
    """
    points = np.asarray(points)
    N = len(points)
    if N < 2:
        raise ValueError("Need at least 2 points.")

    if area is None:
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        area = (xmax - xmin) * (ymax - ymin)

    lambda_density = (N / area)**(-1)
    dists = distance_matrix(points, points)
    np.fill_diagonal(dists, np.inf)

    L_r = []
    for r in radii:
        count = np.sum(dists <= r)
        L = math.sqrt(((lambda_density*(count/N))/math.pi))
        L_r.append(L)
    return np.array(L_r)

radii = np.linspace(0.1, 40 / 2, 50)

test_citizen = []
test_police = []
# Run the model multiple times to collect a base line of the ripley L-function
# for the citizen and police agents.
for i in range (100):
    print(i)
    model = EpsteinCivilViolence(prob_quiet=0.0, reversion_rate=0.0, max_legitimacy_gap=0.0, repression_sensitivity=0.0, max_iters=100)
    model.run_model()
    citizen = model.datacollector.model_vars["citizen"][-1] 
    police = model.datacollector.model_vars["police"][-1]

    print("ripley")
    test_citizen.append(ripley_l_function(points=citizen, radii=radii, area=40*40).tolist())
    test_police.append(ripley_l_function(points=police, radii=radii, area=40*40).tolist())

json.dump(test_citizen, open("./processed/test_citizen.json", "w"))
json.dump(test_police, open("./processed/test_police.json", "w"))

def ripley_apply(series):
    """
    Apply the Ripley L-function to a series of agent coordinates.
    
    Parameters
    ----------
    series : str
        A JSON string representing a list of agent coordinates in the form of
        `[[x1, y1], [x2, y2], ...]`.
    
    Returns
    -------
    list
        A list of values representing the Ripley L-function at the default radii.
    """
    return ripley_l_function(json.loads(series), radii, 40*40).tolist()

# A set of files from which to read the agent coordinates.
# The files are expected to be in the format of a CSV file with a column named "police"
# and a column named "citizen", where each cell contains a JSON string of agent coordinates.
# The output will be saved in a directory named "processed" with the same file name.
files = ["0002_data_0000_0.csv", "0002_data_0000_1.csv", "0002_data_0001_0.csv", "0002_data_0001_1.csv", "0002_data_0002.csv"]
if not os.path.isdir("processed"):
    os.mkdir("processed")
for file in files:
    print(f"Processing {file}")
    filename = f"./outputs_location/{file}"
    data = pd.read_csv(filename)
    data["police"] = data["police"].apply(ripley_apply)
    data["citizen"] = data["citizen"].apply(ripley_apply)
    data.to_csv(f"./processed/{file}")