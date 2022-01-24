from fileinput import filename
from tkinter import N
from turtle import st
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import trange

class PokerAgent:
    def __init__(self, name, env, gamma=0.8, alpha=1e-1,
                 start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999):
        self.env = env
        self.name = name
        self.n_action = self.env.action_space.n
        self.gamma = gamma
        self.alpha = alpha

        # action values
        self.q = defaultdict(lambda: np.zeros(self.n_action))

        # epsilon greedy parameters
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay = epsilon_decay

        # max values during training
        self.max_score = [float("-inf"), float("-inf"), float("-inf"), float("-inf")]
        self.round_max_score = [-1, -1, -1, -1]

    # get epsilon
    def get_epsilon(self, no_episode):
        epsilon = max(self.start_epsilon * (self.epsilon_decay ** no_episode), self.end_epsilon)
        return (epsilon)

    # select action based on epsilon greedy
    def select_action(self, state, epsilon):
        # current legal moves
        legal_moves = [x.value for x in self.env.legal_moves]

        # implicit policy: 
        # if we have action values for that state, choose the largest one, else random
        best_action = legal_moves[np.argmax(self.q[state][legal_moves])] if state in self.q else np.random.choice(legal_moves)
        if np.random.rand() > epsilon:
            action = best_action
        else:
            if legal_moves:
                action = np.random.choice(legal_moves)
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

                if average_score > self.max_score[0]:
                    self.max_score[0] = average_score
                    self.round_max_score[0] = episode_index
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

        print(self.max_score[0], self.round_max_score[0])
        self.save_q_table()

    # saving the current q table in file
    def save_q_table(self):
        filepath = 'data/q_table_' + self.name + '.txt'
        f = open(filepath, "w")

        for key, value in self.q.items():
            value = [x for x in value]
            f.write("{}: {}\n".format(key, value))

        f.close() 

    # loading the q table from file
    def load_q_table(self):
        filepath = 'data/q_table_' + self.name + '.txt'
        f = open(filepath, "r")

        item = f.readline()
        while item:
            if item != '\n':
                item = item.split(": ")
                key = (float(x) for x in item[0][1 : len(item[0]) - 1].split(", "))
                value = [float(x) for x in item[1][1 : len(item[1]) - 3].split(", ")]
                self.q[key] = value
                
            item = f.readline()

        f.close() 