import numpy as np

# Define the game show MDP


class GameShowMDP:
    def __init__(self, rewards, probabilities):
        self.rewards = rewards
        self.probabilities = probabilities
        self.num_questions = len(rewards)
        self.num_states = self.num_questions + 1
        self.start_state = 0
        self.end_state = self.num_states - 1

    def get_transition_probabilities(self, state, action):
        if state == self.end_state:
            return [(self.end_state, 1.0)]
        elif action == 1:
            return [(state + 1, self.probabilities[state]), (self.end_state, 1 - self.probabilities[state])]
        else:
            return [(self.end_state, 1.0)]

    def get_rewards(self, state, action, next_state):
        if next_state == self.end_state:
            return 0
        elif action == 1:
            return self.rewards[state]
        else:
            return 0


def value_iteration(mdp, gamma=1.0, theta=1e-6):
    # Initialize the value function to zero
    V = np.zeros(mdp.num_states)

    while True:
        # Initialize the change in V to zero
        delta = 0

        # Update the value function for each state
        for s in range(mdp.num_states):
            v = V[s]
            V[s] = max([sum([p * (mdp.get_rewards(s, a, next_s) + gamma * V[next_s])
                             for (next_s, p) in mdp.get_transition_probabilities(s, a)])
                        for a in range(2)])
            delta = max(delta, abs(v - V[s]))

        # Check for convergence
        if delta < theta:
            break

    return V
