from poker_agent_general import PokerAgent
import numpy as np

class QLearning_PokerAgent(PokerAgent):
    def __init__(self, env, gamma=0.8, alpha=1e-1,
                 start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999):
        super().__init__('q_learning', env, gamma, alpha, start_epsilon, end_epsilon, epsilon_decay)

    # given (state, action, reward, next_state) pair after a transition made in the environment and the episode index
    def update_experience(self, state, action, reward, next_state):
        best_actionIndex_from_next_state = np.argmax(self.q[next_state])
        best_actionIndex_from_next_state = self.q[next_state][best_actionIndex_from_next_state]

        target = reward + self.gamma * best_actionIndex_from_next_state
        self.q[state][action] = self.q[state][action] + (self.alpha * (target - self.q[state][action]))
