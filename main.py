import sys
import pandas as pd
import numpy as np

from gym_env.env import HoldemTable
from gym_env.env import PlayerShell
from tools.helper import get_config

from agents.agent_random import Player as RandomPlayer
from poker_agent_dqn import DQN_PokerAgent, MethodToUse
from poker_agent_expected_sarsa import ExpectedSarsa_PokerAgent
from poker_agent_q_learning import QLearning_PokerAgent
from poker_agent_sarsa import Sarsa_PokerAgent


"""
Usage:
  main.py [option]

option:
    sarsa                   -- 5 random players + 1 SARSA player
    expected_sarsa          -- 5 random players + 1 Expected SARSA player
    q_learning              -- 5 random players + 1 Q Learning player
    dqn                     -- 5 random players + 1 DQN_BASE player
    dqn_target              -- 5 random players + 1 DQN_TARGET_NETWORK player
    dqn_experience_replay   -- 5 random players + 1 DQN_TARGET_NETWORK_AND_EXPERIENCE_REPLAY player
    dqn_all                 -- 5 random players + 1 DQN_AND_ALL player
    all_basic               -- 1 SARSA player + 1 Expected SARSA player + 1 Q Learning player
    all_dqn                 -- all 4 dqn players
    all                     -- 1 Expected SARSA player + 1 Q Learning player + all 4 dqn players
"""


class SelfPlay:
    """Orchestration of playing against itself"""

    def __init__(self, render, num_episodes, use_cpp_montecarlo, funds_plot, stack=500):
        self.winner_in_episodes = []
        self.use_cpp_montecarlo = use_cpp_montecarlo
        self.funds_plot = funds_plot
        self.render = render
        self.env = None
        self.num_episodes = num_episodes
        self.stack = stack

    def sarsa_agent(self):
        """Create an environment with 5 random players and a sarsa player"""

        self.env = HoldemTable(initial_stacks=self.stack, render=self.render)
        for _ in range(5):
            player = RandomPlayer()
            self.env.add_player(player)
        self.env.add_player(PlayerShell(name='SARSA', stack_size=self.stack))

        self.env.reset()

        sarsaAgent = Sarsa_PokerAgent(self.env, gamma=0.8, alpha=1e-1,
                        start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999)

        scores = []
        actions_per_ep = []

        for _ in range(self.num_episodes):
            self.env.reset()

            score, actions = sarsaAgent.play(no_episodes=1)
            scores.append(score)
            actions_per_ep.append(actions)

            self.winner_in_episodes.append(self.env.winner_ix)

        average_score = np.mean(scores)
        average_actions = np.mean(actions_per_ep)

        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print("Player - No episodes won")
        print(league_table)
        print(f"Best Player: {best_player}")
        print('------------')
        print(f'Average score: {average_score}, Average actions per ep: {average_actions}')

    def expected_sarsa_agent(self):
        """Create an environment with 5 random players and a expected sarsa player"""

        self.env = HoldemTable(initial_stacks=self.stack, render=self.render)
        for _ in range(5):
            player = RandomPlayer()
            self.env.add_player(player)
        self.env.add_player(PlayerShell(name='Expected_SARSA', stack_size=self.stack))

        self.env.reset()
        
        expectedSarsaAgent = ExpectedSarsa_PokerAgent(self.env, gamma=0.8, alpha=1e-1,
                                start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999)

        scores = []
        actions_per_ep = []

        for _ in range(self.num_episodes):
            self.env.reset()

            score, actions = expectedSarsaAgent.play(no_episodes=1)
            scores.append(score)
            actions_per_ep.append(actions)

            self.winner_in_episodes.append(self.env.winner_ix)

        average_score = np.mean(scores)
        average_actions = np.mean(actions_per_ep)

        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print("Player - No episodes won")
        print(league_table)
        print(f"Best Player: {best_player}")
        print('------------')
        print(f'Average score: {average_score}, Average actions per ep: {average_actions}')

    def q_learning_agent(self):
        """Create an environment with 5 random players and a q learning player"""

        self.env = HoldemTable(initial_stacks=self.stack, render=self.render)
        for _ in range(5):
            player = RandomPlayer()
            self.env.add_player(player)
        self.env.add_player(PlayerShell(name='Q_Learning', stack_size=self.stack))

        self.env.reset()
        
        QLearningAgent = QLearning_PokerAgent(self.env, gamma=0.8, alpha=1e-1,
                                start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999)

        scores = []
        actions_per_ep = []

        for _ in range(self.num_episodes):
            self.env.reset()

            score, actions = QLearningAgent.play(no_episodes=1)
            scores.append(score)
            actions_per_ep.append(actions)

            self.winner_in_episodes.append(self.env.winner_ix)

        average_score = np.mean(scores)
        average_actions = np.mean(actions_per_ep)

        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print("Player - No episodes won")
        print(league_table)
        print(f"Best Player: {best_player}")
        print('------------')
        print(f'Average score: {average_score}, Average actions per ep: {average_actions}')

    def get_dqn_agent(self, method):
        if method == MethodToUse.DQN_BASE:
            return DQN_PokerAgent(self.env, seed=42, gamma=0.99, batch_size=64, lr=0.0007,
                      steps_until_sync=200, replay_buffer_size=32000, pre_train_steps=0,
                      start_epsilon = 1, end_epsilon = 0.1, final_epsilon_step = 10000,
                      method=MethodToUse.DQN_BASE)
        
        elif method == MethodToUse.DQN_TARGET_NETWORK:
            return DQN_PokerAgent(self.env, seed=42, gamma=0.99, batch_size=64, lr=0.0007,
                      steps_until_sync=200, replay_buffer_size=32000, pre_train_steps=0,
                      start_epsilon = 1, end_epsilon = 0.1, final_epsilon_step = 10000,
                      method=MethodToUse.DQN_TARGET_NETWORK)

        elif method == MethodToUse.DQN_TARGET_NETWORK_AND_EXPERIENCE_REPLAY:
            return DQN_PokerAgent(self.env, seed=42, gamma=0.99, batch_size=64, lr=0.0007,
                      steps_until_sync=200, replay_buffer_size=32000, pre_train_steps=0,
                      start_epsilon = 1, end_epsilon = 0.1, final_epsilon_step = 10000,
                      method=MethodToUse.DQN_TARGET_NETWORK_AND_EXPERIENCE_REPLAY)

        else:
            return DQN_PokerAgent(self.env, seed=42, gamma=0.99, batch_size=64, lr=0.0007,
                      steps_until_sync=200, replay_buffer_size=32000, pre_train_steps=0,
                      start_epsilon = 1, end_epsilon = 0.1, final_epsilon_step = 10000,
                      method=MethodToUse.DDQN_AND_ALL)

    def dqn_agent(self, method):
        """Create an environment with 5 random players and a dqn player"""

        self.env = HoldemTable(initial_stacks=self.stack, render=self.render)
        for _ in range(5):
            player = RandomPlayer()
            self.env.add_player(player)
        self.env.add_player(PlayerShell(name=method.name, stack_size=self.stack))

        self.env.reset()
        
        DQNAgent = self.get_dqn_agent(method=method)

        scores = []
        actions_per_ep = []

        for _ in range(self.num_episodes):
            self.env.reset()

            score, actions = DQNAgent.play(no_episodes=1)
            scores.append(score)
            actions_per_ep.append(actions)

            self.winner_in_episodes.append(self.env.winner_ix)

        average_score = np.mean(scores)
        average_actions = np.mean(actions_per_ep)

        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print("Player - No episodes won")
        print(league_table)
        print(f"Best Player: {best_player}")
        print('------------')
        print(f'Average score: {average_score}, Average actions per ep: {average_actions}')

    def all_basic_agents(self):
        """Create an environment with all 3 basic players"""

        self.env = HoldemTable(initial_stacks=self.stack, render=self.render)
        self.env.add_player(PlayerShell(name='SARSA', stack_size=self.stack))
        self.env.add_player(PlayerShell(name='Expected_SARSA', stack_size=self.stack))
        self.env.add_player(PlayerShell(name='Q_Learning', stack_size=self.stack))

        self.env.reset()

        sarsaAgent = Sarsa_PokerAgent(self.env, gamma=0.8, alpha=1e-1,
                    start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999)
        sarsaAgent.load_q_table()

        expectedSarsaAgent = ExpectedSarsa_PokerAgent(self.env, gamma=0.8, alpha=1e-1,
                                start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999)
        expectedSarsaAgent.load_q_table()

        QLearningAgent = QLearning_PokerAgent(self.env, gamma=0.8, alpha=1e-1,
                                start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999)
        QLearningAgent.load_q_table()

        for _ in range(self.num_episodes):
            self.env.reset()

            sarsaAgent.play(no_episodes=1)

            self.winner_in_episodes.append(self.env.winner_ix)


        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print("Player - No episodes won")
        print(league_table)
        print(f"Best Player: {best_player}")

    def all_dqn_agents(self):
        """Create an environment with all 4 dqn players"""

        self.env = HoldemTable(initial_stacks=self.stack, render=self.render)
        self.env.add_player(PlayerShell(name='DQN_BASE', stack_size=self.stack))
        self.env.add_player(PlayerShell(name='DQN_TARGET_NETWORK', stack_size=self.stack))
        self.env.add_player(PlayerShell(name='DQN_TARGET_NETWORK_AND_EXPERIENCE_REPLAY', stack_size=self.stack))
        self.env.add_player(PlayerShell(name='DDQN_AND_ALL', stack_size=self.stack))

        self.env.reset()

        DQNAgent1 = self.get_dqn_agent(method=MethodToUse.DQN_BASE)
        DQNAgent1.load_model()

        DQNAgent2 = self.get_dqn_agent(method=MethodToUse.DQN_TARGET_NETWORK)
        DQNAgent2.load_model()

        DQNAgent3 = self.get_dqn_agent(method=MethodToUse.DQN_TARGET_NETWORK_AND_EXPERIENCE_REPLAY)
        DQNAgent3.load_model()

        DQNAgent4 = self.get_dqn_agent(method=MethodToUse.DDQN_AND_ALL)
        DQNAgent4.load_model()

        for _ in range(self.num_episodes):
            self.env.reset()

            DQNAgent1.play(no_episodes=1)

            self.winner_in_episodes.append(self.env.winner_ix)


        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print("Player - No episodes won")
        print(league_table)
        print(f"Best Player: {best_player}")

    def all_agents(self):
        """Create an environment with all 6 players"""

        self.env = HoldemTable(initial_stacks=self.stack, render=self.render)
        self.env.add_player(PlayerShell(name='Expected_SARSA', stack_size=self.stack))
        self.env.add_player(PlayerShell(name='Q_Learning', stack_size=self.stack))
        self.env.add_player(PlayerShell(name='DQN_BASE', stack_size=self.stack))
        self.env.add_player(PlayerShell(name='DQN_TARGET_NETWORK', stack_size=self.stack))
        self.env.add_player(PlayerShell(name='DDQN_AND_ALL', stack_size=self.stack))
        self.env.add_player(PlayerShell(name='DQN_TARGET_NETWORK_AND_EXPERIENCE_REPLAY', stack_size=self.stack))

        self.env.reset()

        expectedSarsaAgent = ExpectedSarsa_PokerAgent(self.env, gamma=0.8, alpha=1e-1,
                                start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999)
        expectedSarsaAgent.load_q_table()

        QLearningAgent = QLearning_PokerAgent(self.env, gamma=0.8, alpha=1e-1,
                                start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999)
        QLearningAgent.load_q_table()

        DQNAgent1 = self.get_dqn_agent(method=MethodToUse.DQN_BASE)
        DQNAgent1.load_model()

        DQNAgent2 = self.get_dqn_agent(method=MethodToUse.DQN_TARGET_NETWORK)
        DQNAgent2.load_model()

        DQNAgent3 = self.get_dqn_agent(method=MethodToUse.DQN_TARGET_NETWORK_AND_EXPERIENCE_REPLAY)
        DQNAgent3.load_model()

        DQNAgent4 = self.get_dqn_agent(method=MethodToUse.DDQN_AND_ALL)
        DQNAgent4.load_model()

        for _ in range(self.num_episodes):
            self.env.reset()

            expectedSarsaAgent.play(no_episodes=1)

            self.winner_in_episodes.append(self.env.winner_ix)


        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print("Player - No episodes won")
        print(league_table)
        print(f"Best Player: {best_player}")


def command_line_parser():
    args = sys.argv[1]
    _ = get_config()

    num_episodes = 3
    runner = SelfPlay(render=True, num_episodes=num_episodes, use_cpp_montecarlo=False,
                      funds_plot=True, stack=20)

    if args == 'sarsa':
        runner.sarsa_agent()

    elif args == 'expected_sarsa':
        runner.expected_sarsa_agent()

    elif args == 'q_learning':
        runner.q_learning_agent()
    
    elif args == 'dqn':
        runner.dqn_agent(method=MethodToUse.DQN_BASE)

    elif args == 'dqn_target':
        runner.dqn_agent(method=MethodToUse.DQN_TARGET_NETWORK)

    elif args == 'dqn_experience_replay':
        runner.dqn_agent(method=MethodToUse.DQN_TARGET_NETWORK_AND_EXPERIENCE_REPLAY)

    elif args == 'dqn_all':
        runner.dqn_agent(method=MethodToUse.DDQN_AND_ALL)   

    elif args == 'all_basic':
        runner.all_basic_agents()

    elif args == 'all_dqn':
        runner.all_dqn_agents()

    elif args == 'all':
        runner.all_agents()

    else:
        raise RuntimeError("Argument not yet implemented")


if __name__ == '__main__':
    command_line_parser()