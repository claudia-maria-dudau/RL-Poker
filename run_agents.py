from poker_agent_sarsa import Sarsa_PokerAgent
from poker_agent_q_learning import QLearning_PokerAgent
from poker_agent_expected_sarsa import ExpectedSarsa_PokerAgent
from poker_agent_dqn import DQN_PokerAgent, MethodToUse
from gym_env.env import HoldemTable
from gym_env.env import PlayerShell
from agents.agent_random import Player as RandomPlayer
from agents.agent_consider_equity import Player as EquityPlayer

def run_sarsa_agent():
    env = HoldemTable()
    env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=.3))
    env.add_player(PlayerShell(name='SARSA', stack_size=500))
    env.reset()

    sarsaAgent = Sarsa_PokerAgent(env, gamma=0.8, alpha=1e-1,
                        start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999)
    sarsaAgent.train(no_episodes=101)

def run_qlearning_agent():
    env = HoldemTable()
    env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=.3))
    env.add_player(PlayerShell(name='Q_Learning', stack_size=500))
    env.reset()

    QLearningAgent = QLearning_PokerAgent(env, gamma=0.8, alpha=1e-1,
                        start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999)
    QLearningAgent.train(no_episodes=101)

def run_expected_sarsa_agent():
    env = HoldemTable()
    env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=.3))
    env.add_player(PlayerShell(name='Expected_SARSA', stack_size=500))
    env.reset()

    ExpectedSarsaAgent = ExpectedSarsa_PokerAgent(env, gamma=0.8, alpha=1e-1,
                            start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999)
    ExpectedSarsaAgent.train(no_episodes=1000)

def run_dqn_agent():
    env = HoldemTable()
    env.add_player(RandomPlayer())
    env.add_player(PlayerShell(name='DQN', stack_size=500))
    env.reset()

    DQNAgent = DQN_PokerAgent(env, seed=42, gamma=0.99, batch_size=64, lr=0.0007,
                      steps_until_sync=200, replay_buffer_size=32000, pre_train_steps=100,
                      start_epsilon = 1, end_epsilon = 0.1, final_epsilon_step = 10000,
                      method=MethodToUse.DQN_TARGET_NETWORK)
    DQNAgent.train(max_episodes=1000, max_steps_per_episode=200, reward_threshold=400, no_episodes_for_average = 10)

# run_sarsa_agent()
# run_qlearning_agent()
# run_expected_sarsa_agent()
run_dqn_agent()