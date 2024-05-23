import os, sys
import gym, h5py
import numpy as np
import random
import json
from tqdm import tqdm
import pandas as pd
np.set_printoptions(precision=3)
import logging, datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
print(f"{'='*140}\n GPU: {tf.config.list_physical_devices('GPU')}\n{'='*140}")

from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.experiments.src.env_utils import create_env, run_episode
from gym_collision_avoidance.envs import test_cases as tc
from GA3C_config import GA3C_config

start = datetime.datetime.now()
# NUM_AGENTS = list(range(2,4+1))
NUM_AGENTS = [4] # list
NUM_ACTIONS = 11
DATASET = 'expert'
NUM_TESTS = 500
POLICIES = 'GA3C_CADRL'
SAVE = True


checkpt_dir = GA3C_config[NUM_ACTIONS][DATASET]['checkpt_dir']
checkpt_name = GA3C_config[NUM_ACTIONS][DATASET]['checkpt_name']
Config.MAX_NUM_AGENTS_IN_ENVIRONMENT = NUM_AGENTS[-1]
Config.MAX_NUM_AGENTS_TO_SIM = NUM_AGENTS[-1]
Config.MAX_NUM_OTHER_AGENTS_OBSERVED = NUM_AGENTS[-1] - 1
Config.TRAIN_SINGLE_AGENT = False
Config.EVALUATE_MODE = True
Config.SAVE_EPISODE_PLOTS = SAVE
Config.SHOW_EPISODE_PLOTS = False
Config.ANIMATE_EPISODES = False
Config.PLOT_CIRCLES_ALONG_TRAJ = True
Config.setup_obs()
Config.MAX_TIME_RATIO = 5

env = create_env()

print(f'\nNUM AGENTS: {NUM_AGENTS}    |   POLICY: {POLICIES} - {DATASET}   |   NUM TESTS: {NUM_TESTS}')
print(f'OBS: {env.observation_space}   |   ACT: {env.action_space}\n\n{"="*140}')

### Configure directory for saving plots and dataset ###
save_dir = os.path.dirname(os.path.realpath(__file__)) + f'/TEST/{NUM_AGENTS[-1]}_agent_{POLICIES}_{DATASET}/'
env.unwrapped.plot_policy_name = POLICIES
env.set_plot_save_dir(save_dir + 'figs/')

ep = samples = total_reward = ep_len = agent_count = 0
results = dict()

test_cases = pd.read_pickle(
    os.path.dirname(os.path.realpath(__file__)) 
    + f'/gym_collision_avoidance/envs/test_cases/{NUM_AGENTS[0]}_agents_500_cases.p'
)
test_stats = {'Tests': NUM_TESTS, 'Policies': POLICIES}
test_stats['summary'] = {}
test_stats['summary'] = {'length': 0, 'reward': 0, 'goal': 0}

for ep in tqdm(range(NUM_TESTS)):
    ### Get agents, initialize policy network (GA3C), set to env ###
    if ep > 500:
        agents = tc.cadrl_test_case_to_agents(test_case=test_cases[ep-500], policies=POLICIES)
    else:
        agents = tc.cadrl_test_case_to_agents(test_case=test_cases[ep], policies=POLICIES)
    [
        agent.policy.initialize_network(checkpt_dir=checkpt_dir, checkpt_name=checkpt_name)
        for agent in agents
        if hasattr(agent.policy, "initialize_network")
    ]
    env.set_agents(agents)
    obs, _ = env.reset()

    env.unwrapped.test_case_index = ep
    ep += 1
    done = False
    ep_length = 0; ep_reward = 0
    ### Run an episode
    while not done:
        obs, rew, done, _, stats = env.step([None])   # Get actions from network
        agent_ts = np.array([(1-int(x)) for x in stats['which_agents_done'].values()])  

        total_reward += np.sum(rew)
        ep_length += agent_ts
        ep_reward += rew

 
    ep_goal = np.array([a.is_at_goal for a in env.agents])
    # Save episodic stats
    test_stats[ep] = {'length': ep_length.tolist(), 'reward': ep_reward.tolist(), 'goal': ep_goal.tolist()}
    # Save total stats
    test_stats['summary']['length'] += ep_length/NUM_TESTS
    test_stats['summary']['reward'] += ep_reward/NUM_TESTS
    test_stats['summary']['goal'] += ep_goal/NUM_TESTS

    env.reset()

test_stats['summary']['average'] = {}
for x in ['length', 'reward', 'goal']:
    test_stats['summary']['average'][x] = np.mean(test_stats['summary'][x]).tolist()
    test_stats['summary'][x] = test_stats['summary'][x].tolist()


if SAVE:
    # Write stats
    with open(save_dir + 'stats.json', 'w') as f:
        f.write(json.dumps(test_stats, indent=4))