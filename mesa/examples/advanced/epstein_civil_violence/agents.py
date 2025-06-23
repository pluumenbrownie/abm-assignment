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
    """
    def __init__(self, model, random_move):
        """
        Create a new EpsteinAgent.
        Args:
            model: the model to which the agent belongs
            random_move: whether to use the logit model for movement.
        """
        super().__init__(model)
        self.random_move = random_move

    def update_neighbors(self):
        """
        Look around and see who my neighbors are
        """
        self.neighborhood = self.cell.get_neighborhood(radius=self.vision)
        self.neighbors = self.neighborhood.agents
        self.empty_neighbors = [c for c in self.neighborhood if c.is_empty]

    def move(self, prob_quite=0.0):
        if self.model.movement and self.empty_neighbors:
            if self.random_move:
                # Randomly choose an empty neighbor
                new_pos = self.random.choice(self.empty_neighbors)
                self.move_to(new_pos)
                return
            values = []
            for neighbor in self.empty_neighbors:
                agents = neighbor.neighborhood.agents
                quiet_count = 0
                for agent in agents:
                    if isinstance(agent, Citizen) and agent.state == CitizenState.QUIET and agent.cell != self.cell:
                        quiet_count += 1
                quiet_count += 1
                values.append(prob_quite/quiet_count)
            if sum(values) == 0:
                # If all rebel rates are zero, we randomly choose an empty neighbor
                new_pos = self.random.choice(self.empty_neighbors)
                self.move_to(new_pos)
                return
            values = np.divide(values, sum(values))
            values = np.cumsum(values)
            new_pos = None

            # If all rebel rates are zero, we randomly choose an empty neighbor
            if values[-1] == 0:
                new_pos = self.random.choice(self.empty_neighbors)
            else:
                new_pos = self.random.choices(self.empty_neighbors, weights=values)[0]
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
        self,
        model,
        threshold,
        vision,
        arrest_prob_constant,
        random_move,
        prob_quiet,
        reversion_rate=0.5,
        max_legitimacy_gap=0.1,
        repression_sensitivity=0.5
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
            model: model instance.
            random_move: whether to use the logit model for movement.
        """
        super().__init__(model, random_move)
        self.hardship = self.random.random()
        self.risk_aversion = self.random.random()
        self.regime_legitimacy = self.random.random()
        self.original_legitimacy = self.regime_legitimacy
        self.threshold = threshold
        self.state = CitizenState.QUIET
        self.vision = vision
        self.jail_sentence = 0
        self.grievance = self.hardship * (1 - self.regime_legitimacy)
        self.arrest_prob_constant = arrest_prob_constant
        self.arrest_probability = None
        self.prob_quiet = prob_quiet

        self.max_legitimacy_gap = max_legitimacy_gap # how much we allow legitimacy to drop
        self.reversion_rate = reversion_rate # the rate at which legitimacy returns to baseline
        self.repression_sensitivity = repression_sensitivity # 1 very resilient, 0 very sensitive

        self.neighborhood = []
        self.neighbors = []
        self.empty_neighbors = []

    def step(self):
        """
        Decide whether to activate, then move if applicable.
        """
        if self.jail_sentence:
            self.jail_sentence -= 1
            return  # no other changes or movements if agent is in jail.

        self.update_neighbors()
        self.update_estimated_arrest_probability()
        self.update_observed_violence()
        self.move(self.prob_quiet)

        net_risk = self.risk_aversion * self.arrest_probability
        if (self.grievance - net_risk) > self.threshold:
            self.state = CitizenState.ACTIVE
        else:
            self.state = CitizenState.QUIET

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
                actives_in_vision += (1-self.prob_quiet)
            elif neighbor.state == CitizenState.QUIET:
                actives_in_vision += self.prob_quiet

        # there is a body of literature on this equation
        # the round is not in the pnas paper but without it, its impossible to replicate
        # the dynamics shown there.
        self.arrest_probability = 1 - math.exp(
            -1 * self.arrest_prob_constant * round(cops_in_vision / actives_in_vision)
        )

    def update_observed_violence(self):
        """
        Return the number of arrested agents in the neighborhood.
        """
        arrests_in_vision = 0
        for neighbor in self.neighbors:
            if neighbor.state == CitizenState.ARRESTED:
                arrests_in_vision += 1

        baseline = self.original_legitimacy
        max_gap = self.max_legitimacy_gap
        alpha = self.reversion_rate

        # Minimum allowable legitimacy
        min_legitimacy = baseline * (1 - max_gap)
        
        if arrests_in_vision > 0:
        # Target a drop toward min_legitimacy, proportional to arrests
        # The more arrests, the closer the target is to the floor
            scaling = 0.1 + (1.0 - self.repression_sensitivity) * 0.9  # ensures >0 division
            decay_fraction = min(1.0, arrests_in_vision / (scaling * 10))
            target = baseline - decay_fraction * (baseline - min_legitimacy)
        else:
        # No arrests → recover toward baseline
            target = baseline

    # Exponential approach to target
        self.regime_legitimacy += alpha * (target - self.regime_legitimacy)

class Cop(EpsteinAgent):
    """
    A cop for life.  No defection.
    Summary of rule: Inspect local vision and arrest a random active agent.

    Attributes:
        unique_id: unique int
        x, y: Grid coordinates
        vision: number of cells in each direction (N, S, E and W) that cop is
            able to inspect
        prob_quiet: probability of arresting a quite citizen instead of an active one.
    """

    def __init__(self, model, vision, max_jail_term, prob_quiet):
        """
        Create a new Cop.
        Args:
            x, y: Grid coordinates
            vision: number of cells in each direction (N, S, E and W) that
                agent can inspect. Exogenous.
            model: model instance
        """
        super().__init__(model, True)
        self.vision = vision
        self.max_jail_term = max_jail_term
        self.prob_quiet = prob_quiet

    def step(self):
        """
        Inspect local vision and arrest a random active agent. Move if
        applicable.
        """
        self.update_neighbors()

        self.move()

        active_neighbors = []
        quiet_neighbors = []
        for agent in self.neighbors:
            if isinstance(agent, Citizen) and agent.state == CitizenState.ACTIVE:
                active_neighbors.append(agent)
            elif isinstance(agent, Citizen) and agent.state == CitizenState.QUIET:
                quiet_neighbors.append(agent)
        if active_neighbors:
            if self.random.random() < self.prob_quiet and quiet_neighbors:
                # Arrest a random quiet citizen
                arrestee = self.random.choice(quiet_neighbors)
            else:
                # Arrest a random active citizen
                arrestee = self.random.choice(active_neighbors)
            arrestee.jail_sentence = self.random.randint(0, self.max_jail_term)
            arrestee.state = CitizenState.ARRESTED
