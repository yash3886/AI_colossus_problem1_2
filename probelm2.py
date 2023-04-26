import numpy as np
from scipy.stats import norm, poisson


class CovidMDP:
    def __init__(self, num_beds=650, population=[0.38, 0.33, 0.23, 0.06], num_weeks=4,
                 bed_conversion_cost=5000, normal_bed_denial_cost=1000, covid_bed_denial_cost=5000):
        self.num_beds = num_beds
        self.population = np.array(population)
        self.num_weeks = num_weeks
        self.bed_conversion_cost = bed_conversion_cost
        self.normal_bed_denial_cost = normal_bed_denial_cost
        self.covid_bed_denial_cost = covid_bed_denial_cost
        self.start_state = (self.num_beds, np.zeros(len(self.population)), np.zeros(len(self.population)),
                            np.zeros(len(self.population)), np.zeros(len(self.population)))
        self.end_state = None
        self.num_states = 0
        self.num_actions = 0
        self.states = {}
        self.actions = {}
        self.rewards = {}
        self.transition_probs = {}

        # Define the possible states and actions
        for i in range(self.num_beds + 1):
            for j in range(self.num_beds + 1 - i):
                for k in range(self.num_weeks):
                    for l in range(len(self.population)):
                        for m in range(len(self.population)):
                            state = (i, j, k, l, m)
                            self.states[state] = self.num_states
                            self.num_states += 1
        for i in range(self.num_beds + 1):
            for j in range(self.num_weeks):
                action = (i, j)
                self.actions[action] = self.num_actions
                self.num_actions += 1

        # Define the reward function
        for state in self.states:
            for action in self.actions:
                r = 0
                beds_normal, beds_covid, week, pop_before, pop_after = state
                beds_normal_new = beds_normal - action[0]
                beds_covid_new = beds_covid + action[0] + action[1]
                pop_migrated = np.round(
                    norm.rvs(0, 1, size=len(self.population)) * 1000)
                pop_before_new = pop_before + pop_migrated
                pop_after_new = pop_after + pop_migrated
                for p in range(len(self.population)):
                    requests_normal = poisson.rvs(
                        pop_after_new[p] * beds_normal_new / self.num_beds)
                    discharges_normal = poisson.rvs(beds_normal_new)
                    denied_normal = max(requests_normal - discharges_normal, 0)
                    requests_covid = poisson.rvs(
                        pop_after_new[p] * beds_covid_new / self.num_beds)
                    discharges_covid = poisson.rvs(beds_covid_new)
                    denied_covid = max(requests_covid - discharges_covid, 0)
                    r += self.normal_bed_denial_cost * denied_normal + \
                        self.covid_bed_denial_cost * denied_covid + \
                        self.bed_conversion_cost * (action[0] + action[1])
                self.rewards[(self.states[state], self.actions[action])] = -r

        # Define the transition probabilities
        for state in self.states:
            for action in self.actions:
                prob_dict = {}
                beds_normal, beds_covid, week, pop_before, pop_after = state
                beds_normal_new = beds_normal - action[0]
                beds_covid_new = beds_covid + action
