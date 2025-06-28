"""
Change the `max_steps`, `distinct_samples` and `param_tuples` in this file to
configure the global sensitivity analysis.
"""

from typing import Any

# the length of the simulations used in the GSA
max_steps = 100

# the amount of samples that should be taken per parameter
# 512 is a good baseline value
distinct_samples = 512

# how often to run each parameter sample per run
# this can be kept at one, because the measurement notebook
# can be kept running until enough samples have been collected
replicates = 1


def get_problem() -> dict[str, Any]:
    """
    Get the problem dictionary for the model.

    Returns a dictionary with the fields
        `"num_vars"`: the amount of parameters in the problem.
        `"names"`: the names of the parameters in the model.
        `"bounds"`: the range the parameter gets sampled in.
    """
    # the parameters to test and their testing range
    param_tuples = [
        ("legitimacy", [0.1, 1.0]),
        ("active_threshold", [0.0, 0.9]),
        ("reversion_rate", [0.0, 1.0]),
        ("prob_quiet", [0.0, 0.5]),
    ]

    param_names, param_bounds = zip(*param_tuples)
    param_names = list(param_names)
    param_bounds = list(param_bounds)

    return {
        "num_vars": len(param_names),
        "names": param_names,
        "bounds": param_bounds,
    }
