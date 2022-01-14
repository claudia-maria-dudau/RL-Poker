from poker_agent_sarsa import Sarsa_PokerAgent
from gym_env.env import HoldemTable
from gym_env.env import PlayerShell
from agents.agent_random import Player as RandomPlayer
from agents.agent_consider_equity import Player as EquityPlayer

def run_sarsa_agent():
    env = HoldemTable()
    env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=.3))
    env.add_player(PlayerShell(name='SARSA', stack_size=500))

    sarsaAgent = Sarsa_PokerAgent(env, gamma=0.8, alpha=1e-1,
                        start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999)
    sarsaAgent.train(no_episodes=101)

run_sarsa_agent()