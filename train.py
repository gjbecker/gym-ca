import os, sys
import gym, h5py
import numpy as np
np.set_printoptions(precision=3)
import logging, datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
print(f"{'='*140}\n GPU: {tf.config.list_physical_devices('GPU')}\n{'='*140}")

from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.experiments.src.env_utils import create_env, run_episode
from gym_collision_avoidance.envs import test_cases as tc
from gym_collision_avoidance.envs.policies.GA3C_CADRL.network import Actions, Actions_Plus

start = datetime.datetime.now()
NUM_AGENTS = 4
SAMPLE_TARGET = 1e6
POLICIES = 'GA3C_CADRL'
SAVE = True

checkpt_dir = 'run-20240322_114351-opn3hmir/files/checkpoints'
# checkpt_name = 'network_01000001'
checkpt_name = 'network_00200000'
Config.ACTION_SPACE_TYPE = Config.discrete
Config.ACTIONS = Actions()
actions = Config.ACTIONS.actions
Config.MAX_NUM_AGENTS_IN_ENVIRONMENT = NUM_AGENTS
Config.MAX_NUM_AGENTS_TO_SIM = NUM_AGENTS
Config.MAX_NUM_OTHER_AGENTS_OBSERVED = NUM_AGENTS - 1
Config.SAVE_EPISODE_PLOTS = False
Config.TRAIN_SINGLE_AGENT = False
Config.setup_obs()

env = create_env()

print(f'\nNUM AGENTS: {NUM_AGENTS}    |   POLICY: {POLICIES}   |   SAMPLE TARGET: {SAMPLE_TARGET:,.0f}')
print(f'OBS: {env.observation_space}   |   ACT: {env.action_space}\n\n{"="*140}')

save_dir = os.path.dirname(os.path.realpath(__file__)) + f'/GA3C/{NUM_AGENTS}_agent_{Config.ACTIONS.num_actions}_actions/'
env.unwrapped.plot_policy_name = 'GA3C'
env.set_plot_save_dir(save_dir + 'figs/')

ep = samples = total_reward = ep_len = 0

O, CA, DA, R, T = [{x:[] for x in range(NUM_AGENTS)} for _ in range(5)]
results = dict()

while samples < SAMPLE_TARGET:
    ### Get agents, initialize policy network (GA3C), set to env
    agents = tc.get_testcase_random(num_agents=NUM_AGENTS, policies=POLICIES)
    [
        agent.policy.initialize_network(checkpt_dir=checkpt_dir, checkpt_name=checkpt_name)
        for agent in agents
        if hasattr(agent.policy, "initialize_network")
    ]
    env.set_agents(agents)
    obs, _ = env.reset()
    env.unwrapped.test_case_index = ep
    ep += 1
    step = 0
    terminated = False
    ### Run an episode
    while not terminated:
        next_obs, rew, terminated, _, info = env.step([None])   # Get actions from network
        dones = info['which_agents_done']    
        
        for i, agent in enumerate(env.agents):
            if dones[i] == True and step == 0:
                samples+=1
                T[i].extend([dones[i]])
                O[i].extend([obs[i]])
                R[i].extend([rew[i]])
                CA[i].extend([agent.past_actions[0]])
                for j,a in enumerate(actions):
                    if np.allclose(a, agent.past_actions[0]/[obs[i][4], 1]):
                        DA[i].extend([j])
                        break

            elif dones[i] == False or dones[i] != T[i][-1]:
                samples+=1
                T[i].extend([dones[i]])
                O[i].extend([obs[i]])
                R[i].extend([rew[i]])
                CA[i].extend([agent.past_actions[0]])
                for j,a in enumerate(actions):
                    if np.allclose(a, agent.past_actions[0]/[obs[i][4], 1]):
                        # print(i, agent.past_actions[0]/[obs[i][4], 1], Actions_Plus().actions[j], a)
                        DA[i].extend([j])
                        break

        step+=1
        total_reward += rew
        obs = next_obs
    ep_len += step
    if ep % 5 == 0:
        print(
        f'|    EP: {ep:4d}    ||    AVG LENGTH: {ep_len/ep:.0f}    ||    AVG REWARD: {np.sum(total_reward)/(NUM_AGENTS*ep): .2f}    |'
        f'|    SAMPLES: {samples:7d}    ||    PROGRESS: {(samples/SAMPLE_TARGET):4.2%}    |    {datetime.datetime.now()-start}    |'
        )
    env.reset()

O_ = np.array(O[0]); R_ = np.array(R[0]); T_ = np.array(T[0]); CA_ = np.array(CA[0]); DA_ = np.array(DA[0])

for i in range(1, NUM_AGENTS):
    O_ = np.concatenate((O_, O[i])); R_ = np.concatenate((R_, R[i])); T_ = np.concatenate((T_, T[i]))
    CA_ = np.concatenate((CA_, CA[i])); DA_ = np.concatenate((DA_, DA[i]))

# print(O_.shape,R_.shape,T_.shape, CA_.shape, DA_.shape)

results['observations'] = O_
results['rewards'] = R_
results['terminals'] = T_
results['c_actions'] = CA_
results['d_actions'] = DA_

filename = save_dir + 'dataset.hdf5'

if SAVE:
    file = h5py.File(filename, 'w')
    for name in results:  
        file.create_dataset(name, data=results[name], compression='gzip')

    file['metadata/source'] = f'{checkpt_dir}/{checkpt_name}'
    file['metadata/time'] = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    file.close()

    print(f'{"="*140}\nSAVED DATASET: {filename}\n{"="*140}')