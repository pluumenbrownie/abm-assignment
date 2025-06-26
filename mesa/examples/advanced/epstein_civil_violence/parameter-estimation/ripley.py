import pandas as pd
import json
import numpy as np
from scipy.spatial import distance_matrix
import os
import math
import random
import sys
sys.path.append(os.path.abspath("../../../../.."))
from mesa.examples.advanced.epstein_civil_violence.model import EpsteinCivilViolence

def ripley_l_function(points, radii, area=None):
    """
    Compute Ripley's K-function for a set of 2D points.
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

def initial(series):
    return ripley_l_function(json.loads(series), radii, 40*40).tolist()

files = ["0002_data_0000.csv", "0002_data_0001.csv", "0002_data_0002.csv"]
if not os.path.isdir("processed"):
    os.mkdir("processed")
for file in files:
    filename = f"./outputs/{file}"
    data = pd.read_csv(filename)
    data["police"] = data["police"].apply(initial)
    data["citizen"] = data["citizen"].apply(initial)
    data.to_csv(f"./processed/{file}")