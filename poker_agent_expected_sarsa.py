from poker_agent_general import PokerAgent
import numpy as np

class ExpectedSarsa_PokerAgent(PokerAgent):
    def __init__(self, env, gamma=0.8, alpha=1e-1,
                 start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999):
        super().__init__(env, gamma, alpha, start_epsilon, end_epsilon, epsilon_decay)

    def update_experience(self, state, action, reward, next_state, n_episode):
        # get the action at the next step using e-greedy policy
        next_action = self.select_action(next_state, self.get_epsilon(n_episode))

        # get current epsilon
        eps = self.get_epsilon(n_episode)

        # Remember that we set an epsilon then draw a variable X:
        # - if X < epsilon:
        #       each action has an equal chance of (1/|num actions|)
        #   else:
        #       best action will be selected.
        #
        # get Q value of random portion (X < epsilon)


        prob_actions_whenRandom = np.array([(eps * (1/self.n_action)) for action_index in range(0, self.n_action)])
        prob_actions_whenBest = np.array([0 if action_index != next_action else (1.0-eps) \
                                         for action_index in range(0, self.n_action)])

        # it is a plus because the best action has the equal chance still when X < eps !
        prob_actions = prob_actions_whenRandom + prob_actions_whenBest

        # dot product between actions and their q values
        est_value_from_next_state = np.sum(prob_actions * self.q[next_state])
        target = reward + self.gamma * est_value_from_next_state

        self.q[state][action] = self.q[state][action] + (self.alpha * (target - self.q[state][action]))