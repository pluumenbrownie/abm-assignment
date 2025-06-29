from collections.abc import Callable
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import solara

from mesa.examples.advanced.epstein_civil_violence.agents import (
    Citizen,
    CitizenState,
    Cop,
)
from mesa.examples.advanced.epstein_civil_violence.model import EpsteinCivilViolence
from mesa.visualization import (
    Slider,
    SolaraViz,
    make_plot_component,
    make_space_component,
)
from mesa.visualization.utils import update_counter

COP_COLOR = "#000000"

agent_colors = {
    CitizenState.ACTIVE: "#FE6100",
    CitizenState.QUIET: "#648FFF",
    CitizenState.ARRESTED: "#808080",
}

cmap = mpl.colormaps["cool"]


def color_quiet(agent: Citizen) -> str:
    """Colors the agent according to their grievance."""
    return "#" + "".join([hex(x)[2:] for x in cmap(agent.grievance, bytes=True)])[:6]


def citizen_cop_portrayal(agent: Citizen | Cop) -> dict[str, Any]:
    """
    Gives the matplotlib backend instructions how to draw this agent in the grid.

    Args:
        agent (Citizen | Cop): the agent to draw

    Returns:
        dict: a dictionary with matplotlib drawing kwargs.
    """
    if agent is None:
        return

    portrayal = {
        "size": 50,
    }

    if isinstance(agent, Citizen):
        if agent.state == CitizenState.QUIET:
            portrayal["color"] = color_quiet(agent)
        else:
            portrayal["color"] = agent_colors[agent.state]
    elif isinstance(agent, Cop):
        portrayal["color"] = COP_COLOR

    return portrayal


def post_process(ax):
    """Styles the spatial plot."""
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_figure().set_size_inches(10, 10)


def make_mpl_plot_component_from_function(
    measure: Callable | dict[str, Callable],
    post_process: Callable | None = None,
    save_format="png",
):
    """Create a plotting function for a specified measuring function.

    Args:
        measure (Callable | dict[str, Callable]): Measuring function(s) to plot.
        post_process: a user-specified callable to do post-processing called with the Axes instance.
        save_format: save format of figure in solara backend

    Returns:
        function: A function that creates a PlotMatplotlib component.
    """

    def MakePlotMatplotlib(model):
        return PlotMatplotlibFunction(
            model, measure, post_process=post_process, save_format=save_format
        )

    return MakePlotMatplotlib


@solara.component
def PlotMatplotlibFunction(
    model,
    measure,
    dependencies: list[Any] | None = None,
    post_process: Callable | None = None,
    save_format="png",
):
    """Create a Matplotlib-based plot by applying a function to the model.

    Args:
        model (mesa.Model): The model instance.
        measure (str | dict[str, Callable]): Measure(s) to plot.
        dependencies (list[any] | None): Optional dependencies for the plot.
        post_process: a user-specified callable to do post-processing called with the Axes instance.
        save_format: format used for saving the figure.

    Returns:
        solara.FigureMatplotlib: A component for rendering the plot.
    """
    update_counter.get()
    fig = plt.figure()
    ax = fig.subplots()
    if isinstance(measure, Callable):
        ax.plot(measure(model))
    elif isinstance(measure, dict):
        for name, func in measure.items():
            ax.plot(func(model), label=name)
        ax.legend(loc="best")

    if post_process is not None:
        post_process(ax)

    ax.set_xlabel("Step")
    # Set integer x axiss
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    solara.FigureMatplotlib(
        fig, format=save_format, bbox_inches="tight", dependencies=dependencies
    )


model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "height": 40,
    "width": 40,
    "citizen_density": Slider("Initial Agent Density", 0.7, 0.0, 0.9, 0.1),
    "cop_density": Slider("Initial Cop Density", 0.04, 0.0, 0.1, 0.01),
    "citizen_vision": Slider("Citizen Vision", 7, 1, 10, 1),
    "cop_vision": Slider("Cop Vision", 7, 1, 10, 1),
    "legitimacy": Slider("Government Legitimacy", 0.82, 0.0, 1, 0.01),
    "max_jail_term": Slider("Max Jail Term", 30, 0, 50, 1),
    "prob_quiet": Slider("Quiet Arrest Chance", 0.1, 0.0, 1.0, 0.01),
    "active_threshold": Slider("Active Threshold", 0.1, 0.0, 1.0, 0.01),
    "reversion_rate": Slider("Reversion Rate", 0.05, 0.0, 1.0, 0.01),
    "random_move_agent": {
        "type": "Checkbox",
        "value": False,
        "label": "Random Move Citizens",
    },
}

# the grid containing the locations of the agents
space_component = make_space_component(
    citizen_cop_portrayal, post_process=post_process, draw_grid=False
)

# the evolution of the number of agents per state
chart_component = make_plot_component(
    {state.name.lower(): agent_colors[state] for state in CitizenState}
)

# the Ripley statistic for the model, a measure of clustering
ripley_chart = make_mpl_plot_component_from_function(
    {
        "citizens": lambda m: m.ripley_l_function(
            m.datacollector.model_vars["citizen"][-1],
            np.linspace(0.1, m.grid.width / 2, 50),
            area=40 * 40,
        ),
        "police": lambda m: m.ripley_l_function(
            m.datacollector.model_vars["police"][-1],
            np.linspace(0.1, m.grid.width / 2, 50),
            area=40 * 40,
        ),
    }
)

epstein_model = EpsteinCivilViolence()

page = SolaraViz(
    epstein_model,
    components=[space_component, chart_component, ripley_chart],
    model_params=model_params,
    name="Epstein Civil Violence",
)
page  # noqa
