from tkinter import N
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import trange

class PokerAgent:
    def __init__(self, env, gamma=0.8, alpha=1e-1,
                 start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999):
        self.env = env
        self.n_action = self.env.action_space.n
        self.gamma = gamma
        self.alpha = alpha

        # action values
        self.q = defaultdict(lambda: np.zeros(self.n_action))

        # epsilon greedy parameters
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = epsilon_decay

    # get epsilon
    def get_epsilon(self, n_episode):
        epsilon = max(self.start_epsilon * (self.epsilon_decay ** n_episode), self.end_epsilon)
        return (epsilon)

    # select action based on epsilon greedy
    def select_action(self, state, epsilon):
        # implicit policy; if we have action values for that state, choose the largest one, else random
        best_action = np.argmax(self.q[state]) if state in self.q else self.env.action_space.sample()
        
        if np.random.rand() > epsilon:
            action = best_action
        else:
            if self.env.legal_moves != None:
                action = np.random.choice([x.value for x in self.env.legal_moves])
            else:  
                action = self.env.action_space.sample()
            
        return (action)

    # training the agent
    def train(self, no_episodes, plot_stats=True):
        average_nb = 100

        scores = []
        average_scores = []
        actions_per_ep = []

        for episode_index in trange(no_episodes):
            # for the record
            score = 0
            no_actions = 0

            # initiate state
            state = tuple(self.env.reset())
            while True:
                # get action
                eps_at_episode = self.get_epsilon(episode_index)
                action = self.select_action(state, epsilon=eps_at_episode)

                # step environment
                next_state, reward, done, info = self.env.step(action)
                next_state = tuple(next_state)

                # update agent
                self.update_experience(state, action, reward, next_state, episode_index)

                # records
                score += reward
                no_actions += 1

                # move to next state
                state = next_state

                if done:
                    break

            # record
            scores.append(score)
            actions_per_ep.append(no_actions)

            if episode_index > average_nb:
                average_score = np.mean(scores[episode_index - average_nb:episode_index])
                average_scores.append(average_score)
            else:
                average_scores.append(0)

        if plot_stats:
            plt.title("Scores per episode")
            plt.xlabel("Episode")
            plt.ylabel("Score")
            plt.plot(scores)
            plt.show()
            print(f'Last 100-episode average score at the end of simulation: {average_scores[-1]}')

            plt.title("Number of actions performed per episode")
            plt.xlabel("Episode")
            plt.ylabel("No. of actions")
            plt.plot(actions_per_ep)
            plt.show()
            print(f'Last 100-episode average number of actions performed: {np.mean(actions_per_ep[:-100])}')