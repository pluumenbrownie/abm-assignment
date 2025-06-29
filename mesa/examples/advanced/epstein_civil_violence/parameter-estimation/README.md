# Global sensitivity analysis

The generation of parameter samples, measurement data, and the final analysis are split into separate steps.
How to run GSA:

1. Update settings in `constants.py`.
1. Run `generate_parameters.ipynb`.
1. Run `measure_sensitivity.ipynb` until enough data has been collected.
1. Perform global sensitivity analysis with `GSA-analysis.ipynb`.

# Ripley analysis

How to run Ripley analysis:

1. Run `ripley.py` to generate data.
1. Run `run.py` to generate plots for sensitivity analysis and Ripley data.
