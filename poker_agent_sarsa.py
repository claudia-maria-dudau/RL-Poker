from poker_agent_general import PokerAgent

class Sarsa_PokerAgent(PokerAgent):
    def __init__(self, env, gamma=0.8, alpha=1e-1,
                 start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999):
        super().__init__(env, gamma, alpha, start_epsilon, end_epsilon, epsilon_decay)

    def update_experience(self, state, action, reward, next_state, n_episode):
        #get next action
        next_action = self.select_action(next_state, self.get_epsilon(n_episode))

        #get new q
        new_q = reward + (self.gamma * self.q[next_state][next_action])

        #calculate update equation
        self.q[state][action] = self.q[state][action] + (self.alpha * (new_q - self.q[state][action]))