from model import EpsteinCivilViolence
import numpy as np
import sys
import math
from scipy.spatial import distance_matrix

def ripley_k_function(points, radii, area=None):
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

    lambda_density = N / area
    dists = distance_matrix(points, points)
    np.fill_diagonal(dists, np.inf)

    K_r = []
    for r in radii:
        count = np.sum(dists <= r)
        K = count / lambda_density
        K_r.append(K)
    return np.array(K_r)

def ripley_l_function(points, radii, area=None):
    """
    Compute Ripley's L-function from a set of 2D points.
    """
    K_r = ripley_k_function(points, radii, area)
    L_r = np.sqrt(K_r / np.pi) - radii
    return L_r


np.set_printoptions(threshold=sys.maxsize)
nx = 20
ny = 20

model = EpsteinCivilViolence(
    height=ny,
    width=nx,
    max_jail_term=1,
    max_iters=100,
)  # cap the number of steps the model takes
model.run_model()
model_data = model.datacollector.get_model_vars_dataframe()
police = model_data["police_location"].values
citizen = model_data["citizen_location"].values

for i, row in enumerate(zip(police, citizen)):
    radii = np.linspace(0.1, 10.0, 50)
    ripley_l_police = ripley_l_function(points=np.array(row[0]), radii=radii, area=nx*ny)
    ripley_l_citizen = ripley_l_function(points=np.array(row[1]), radii=radii, area=nx*ny)
    print(f"Step {i}: Ripley's L-function for police: {ripley_l_police}, radii: {radii}")
    print(f"Step {i}: Ripley's L-function for citizen: {ripley_l_citizen}, radii: {radii}")


# @article{lynch2008spatiotemporal,
#   title={A spatiotemporal Ripley’s K-function to analyze interactions between spruce budworm and fire in British Columbia, Canada},
#   author={Lynch, Heather J and Moorcroft, Paul R},
#   journal={Canadian Journal of Forest Research},
#   volume={38},
#   number={12},
#   pages={3112--3119},
#   year={2008}
# }
# @article{giuliani2014weighting,
#   title={Weighting Ripley’s K-function to account for the firm dimension in the analysis of spatial concentration},
#   author={Giuliani, Diego and Arbia, Giuseppe and Espa, Giuseppe},
#   journal={International Regional Science Review},
#   volume={37},
#   number={3},
#   pages={251--272},
#   year={2014},
#   publisher={SAGE Publications Sage CA: Los Angeles, CA}
# }
# @article{haase1995spatial,
#   title={Spatial pattern analysis in ecology based on Ripley's K-function: Introduction and methods of edge correction},
#   author={Haase, Peter},
#   journal={Journal of vegetation science},
#   volume={6},
#   number={4},
#   pages={575--582},
#   year={1995},
#   publisher={Wiley Online Library}
# }
# @article{ripley1977modelling,
#   title={Modelling spatial patterns},
#   author={Ripley, Brian D},
#   journal={Journal of the Royal Statistical Society: Series B (Methodological)},
#   volume={39},
#   number={2},
#   pages={172--192},
#   year={1977},
#   publisher={Wiley Online Library}
# }
