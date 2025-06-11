import math
from enum import Enum

import mesa
import numpy as np

class CitizenState(Enum):
    ACTIVE = 1
    QUIET = 2
    ARRESTED = 3


class EpsteinAgent(mesa.discrete_space.CellAgent):
    """
    Attributes:
        model: model instance
        lamb: (λ) the hyperparameter for the logit model.
    """
    def __init__(self, model, lamb):
        """
        Create a new EpsteinAgent.
        Args:
            model: the model to which the agent belongs
            lamb: (λ) the hyperparameter for the logit model.
        """
        super().__init__(model)
        self.lamb = lamb

    def update_neighbors(self):
        """
        Look around and see who my neighbors are
        """
        self.neighborhood = self.cell.get_neighborhood(radius=self.vision)
        self.neighbors = self.neighborhood.agents
        self.empty_neighbors = [c for c in self.neighborhood if c.is_empty]

    def move(self, rebel_layer: mesa.discrete_space.PropertyLayer, high: bool = False):
        if self.model.movement and self.empty_neighbors:
            # include logit model for movement
            rebel_rate = []
            for neighbor in self.empty_neighbors:
                # We calculate e(λ * rebel_rate) for each empty neighbor
                rebel_rate.append(math.exp(self.lamb * (rebel_layer.data[neighbor.coordinate[0]][neighbor.coordinate[1]])))
            m = max(rebel_rate)
            if not high:
                # If we want to move to a high-rebel area, we invert the rates
                rebel_rate = np.apply_along_axis(lambda x: m-x, 0, rebel_rate)
            if rebel_rate[-1] == 0:
                # If all rebel rates are zero, we randomly choose an empty neighbor
                new_pos = self.random.choice(self.empty_neighbors)
                self.move_to(new_pos)
                return
            # Normalize the rebel rates and calculate cumulative sumq
            rebel_rate = np.divide(rebel_rate, sum(rebel_rate))
            rebel_rate = np.cumsum(rebel_rate)
            new_pos = None

            # If all rebel rates are zero, we randomly choose an empty neighbor
            if rebel_rate[-1] == 0:
                new_pos = self.random.choice(self.empty_neighbors)
            else:
                new_pos = self.random.choices(self.empty_neighbors, weights=rebel_rate)[0]
            self.move_to(new_pos)


class Citizen(EpsteinAgent):
    """
    A member of the general population, may or may not be in active rebellion.
    Summary of rule: If grievance - risk > threshold, rebel.

    Attributes:
        hardship: Agent's 'perceived hardship (i.e., physical or economic
            privation).' Exogenous, drawn from U(0,1).
        regime_legitimacy: Agent's perception of regime legitimacy, equal
            across agents.  Exogenous.
        risk_aversion: Exogenous, drawn from U(0,1).
        threshold: if (grievance - (risk_aversion * arrest_probability)) >
            threshold, go/remain Active
        vision: number of cells in each direction (N, S, E and W) that agent
            can inspect
        condition: Can be "Quiescent" or "Active;" deterministic function of
            greivance, perceived risk, and
        grievance: deterministic function of hardship and regime_legitimacy;
            how aggrieved is agent at the regime?
        arrest_probability: agent's assessment of arrest probability, given
            rebellion
    """

    def __init__(
        self, model, regime_legitimacy, threshold, vision, arrest_prob_constant, lamb
    ):
        """
        Create a new Citizen.
        Args:
            model: the model to which the agent belongs
            hardship: Agent's 'perceived hardship (i.e., physical or economic
                privation).' Exogenous, drawn from U(0,1).
            regime_legitimacy: Agent's perception of regime legitimacy, equal
                across agents.  Exogenous.
            risk_aversion: Exogenous, drawn from U(0,1).
            threshold: if (grievance - (risk_aversion * arrest_probability)) >
                threshold, go/remain Active
            vision: number of cells in each direction (N, S, E and W) that
                agent can inspect. Exogenous.
            model: model instance
            lamb: (λ) the hyperparameter for the logit model.
        """
        super().__init__(model, lamb)
        self.hardship = self.random.random()
        self.risk_aversion = self.random.random()
        self.regime_legitimacy = regime_legitimacy
        self.threshold = threshold
        self.state = CitizenState.QUIET
        self.vision = vision
        self.jail_sentence = 0
        self.grievance = self.hardship * (1 - self.regime_legitimacy)
        self.arrest_prob_constant = arrest_prob_constant
        self.arrest_probability = None

        self.neighborhood = []
        self.neighbors = []
        self.empty_neighbors = []

    def step(self, rebel_layer: mesa.discrete_space.PropertyLayer):
        """
        Decide whether to activate, then move if applicable.
        """
        if self.jail_sentence:
            self.jail_sentence -= 1
            return  # no other changes or movements if agent is in jail.
        self.update_neighbors()
        self.update_estimated_arrest_probability()

        net_risk = self.risk_aversion * self.arrest_probability
        if (self.grievance - net_risk) > self.threshold:
            self.state = CitizenState.ACTIVE
        else:
            self.state = CitizenState.QUIET

        self.move(rebel_layer, False)

    def update_estimated_arrest_probability(self):
        """
        Based on the ratio of cops to actives in my neighborhood, estimate the
        p(Arrest | I go active).
        """
        cops_in_vision = 0
        actives_in_vision = 1  # citizen counts herself
        for neighbor in self.neighbors:
            if isinstance(neighbor, Cop):
                cops_in_vision += 1
            elif neighbor.state == CitizenState.ACTIVE:
                actives_in_vision += 1

        # there is a body of literature on this equation
        # the round is not in the pnas paper but without it, its impossible to replicate
        # the dynamics shown there.
        self.arrest_probability = 1 - math.exp(
            -1 * self.arrest_prob_constant * round(cops_in_vision / actives_in_vision)
        )


class Cop(EpsteinAgent):
    """
    A cop for life.  No defection.
    Summary of rule: Inspect local vision and arrest a random active agent.

    Attributes:
        unique_id: unique int
        x, y: Grid coordinates
        vision: number of cells in each direction (N, S, E and W) that cop is
            able to inspect
        lamb: (λ) the hyperparameter for the logit model.
    """

    def __init__(self, model, vision, max_jail_term, lamb):
        """
        Create a new Cop.
        Args:
            x, y: Grid coordinates
            vision: number of cells in each direction (N, S, E and W) that
                agent can inspect. Exogenous.
            model: model instance
            lamb: (λ) the hyperparameter for the logit model.
        """
        super().__init__(model, lamb)
        self.vision = vision
        self.max_jail_term = max_jail_term

    def step(self, rebel_layer: mesa.discrete_space.PropertyLayer):
        """
        Inspect local vision and arrest a random active agent. Move if
        applicable.
        """
        self.update_neighbors()
        active_neighbors = []
        for agent in self.neighbors:
            if isinstance(agent, Citizen) and agent.state == CitizenState.ACTIVE:
                active_neighbors.append(agent)
        if active_neighbors:
            arrestee = self.random.choice(active_neighbors)
            arrestee.jail_sentence = self.random.randint(0, self.max_jail_term)
            arrestee.state = CitizenState.ARRESTED

        self.move(rebel_layer, True)
