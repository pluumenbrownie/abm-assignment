# ruff: noqa: N809

import numpy as np
from scipy.spatial import distance_matrix

import mesa

# from mesa.examples.advanced.epstein_civil_violence.agents_new import (
#     Citizen,
#     CitizenState,
#     Cop,
# )
from mesa.examples.advanced.epstein_civil_violence.agents import (
    Citizen,
    CitizenState,
    Cop,
)


class EpsteinCivilViolence(mesa.Model):
    """
    Model 1 from "Modeling civil violence: An agent-based computational
    approach," by Joshua Epstein.
    http://www.pnas.org/content/99/suppl_3/7243.full

    Args:
        height: grid height
        width: grid width
        citizen_density: approximate % of cells occupied by citizens.
        cop_density: approximate % of cells occupied by cops.
        citizen_vision: number of cells in each direction (N, S, E and W) that
            citizen can inspect
        cop_vision: number of cells in each direction (N, S, E and W) that cop
            can inspect
        legitimacy:  (L) citizens' perception of regime legitimacy, equal
            across all citizens
        max_jail_term: (J_max)
        active_threshold: if (grievance - (risk_aversion * arrest_probability))
            > threshold, citizen rebels
        arrest_prob_constant: set to ensure agents make plausible arrest
            probability estimates
        movement: binary, whether agents try to move at step end
        max_iters: model may not have a natural stopping point, so we set a
            max.
    """

    def __init__(
        self,
        width=40,
        height=40,
        citizen_density=0.7,
        cop_density=0.074,
        citizen_vision=7,
        cop_vision=7,
        legitimacy=0.8,
        max_jail_term=1000,
        active_threshold=0.1,
        arrest_prob_constant=2.3,
        movement=True,
        max_iters=1000,
        enable_agent_reporters=False,
        seed=None,
        random_move_agent=False,
        prob_quiet=0.1,
        reversion_rate=0.05,
        max_legitimacy_gap=0.5,
        repression_sensitivity=0.5,
    ):
        super().__init__(seed=seed)
        self.movement = movement
        self.max_iters = max_iters

        self.legitimacy = legitimacy
        self.reversion_rate = reversion_rate
        self.max_legitimacy_gap = max_legitimacy_gap
        self.repression_sensitivity = repression_sensitivity

        self.grid = mesa.discrete_space.OrthogonalVonNeumannGrid(
            (width, height), capacity=1, torus=True, random=self.random
        )

        self.radii = np.linspace(0.1, width / 2, 50)
        model_reporters = {
            "active": CitizenState.ACTIVE.name,
            "quiet": CitizenState.QUIET.name,
            "arrested": CitizenState.ARRESTED.name,
            "ripley_l_police": lambda m: m.ripley_l_function(
                points=[
                    [cop.cell.coordinate[0], cop.cell.coordinate[1]]
                    for cop in m.agents_by_type[Cop]
                ],
                radii=m.radii,
                area=width * height,
            ),
            "ripley_l_citizen": lambda m: m.ripley_l_function(
                points=[
                    [cop.cell.coordinate[0], cop.cell.coordinate[1]]
                    for cop in m.agents_by_type[Citizen]
                ],
                radii=m.radii,
                area=width * height,
            ),
        }
        if enable_agent_reporters:
            agent_reporters = {
                "jail_sentence": lambda a: getattr(a, "jail_sentence", None),
                "arrest_probability": lambda a: getattr(a, "arrest_probability", None),
                "regime_legitimacy": lambda a: getattr(a, "regime_legitimacy", None),
                "repression_sensitivity": lambda a: getattr(
                    a, "repression_sensitivity", None
                ),
            }
        else:
            agent_reporters = None
        self.datacollector = mesa.DataCollector(
            model_reporters=model_reporters, agent_reporters=agent_reporters
        )
        if cop_density + citizen_density > 1:
            raise ValueError("Cop density + citizen density must be less than 1")

        for cell in self.grid.all_cells:
            klass = self.random.choices(
                [Citizen, Cop, None],
                cum_weights=[citizen_density, citizen_density + cop_density, 1],
            )[0]

            if klass == Cop:
                cop = Cop(
                    self,
                    vision=cop_vision,
                    max_jail_term=max_jail_term,
                    prob_quiet=prob_quiet,
                )
                cop.move_to(cell)
            elif klass == Citizen:
                citizen = Citizen(
                    self,
                    regime_legitimacy=self.legitimacy,
                    threshold=active_threshold,
                    vision=citizen_vision,
                    arrest_prob_constant=arrest_prob_constant,
                    random_move=random_move_agent,
                    prob_quiet=prob_quiet,
                    reversion_rate=self.reversion_rate,
                    max_legitimacy_gap=self.max_legitimacy_gap,
                    repression_sensitivity=self.repression_sensitivity,
                )
                citizen.move_to(cell)

        self.running = True
        self._update_counts()
        self.datacollector.collect(self)

    def ripley_k_function(self, points, radii, area=None):
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

    def ripley_l_function(self, points, radii, area=None):
        """
        Compute Ripley's L-function from a set of 2D points.
        """
        K_r = self.ripley_k_function(points, radii, area)
        L_r = np.sqrt(K_r / np.pi) - radii
        return L_r

    def step(self):
        """
        Advance the model by one step and collect data.
        """
        self.agents.shuffle_do("step")
        self._update_counts()
        self.datacollector.collect(self)

        if self.steps > self.max_iters:
            self.running = False

    def _update_counts(self):
        """Helper function for counting nr. of citizens in given state."""
        counts = self.agents_by_type[Citizen].groupby("state").count()

        for state in CitizenState:
            setattr(self, state.name, counts.get(state, 0))
