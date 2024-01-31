import os
import pickle
import time
import logging
import configparser
import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ["GYM_CONFIG_CLASS"] = "DataGeneration"
import gym_collision_avoidance.envs.test_cases as tc
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.experiments.src.env_utils import (
    create_env,
    policies,
    run_episode,
    store_stats,
)
rewCONF = configparser.ConfigParser()
rewCONF.read('reward.config')

def reset_env(
    env,
    test_case_fn,
    test_case_args,
    test_case,
    num_agents,
    policies,
    policy,
    prev_agents,
):
    env.unwrapped.plot_policy_name = policy
    test_case_args["num_agents"] = num_agents
    test_case_args["prev_agents"] = prev_agents
    if policy == 'circle':
        rnd = np.random.rand()
        if rnd < 0.3:
            agents = tc.circle_test_case_to_agents(num_agents=4, circle_radius=2.5)
        elif rnd > 0.7:
            agents = tc.circle_test_case_to_agents(num_agents=[4, 4], circle_radius=[2.5, 4.5])
        else:
            agents = tc.circle_test_case_to_agents(num_agents=6, circle_radius=3.5)
    elif policy == 'circle8':
        agents = tc.circle_test_case_to_agents(num_agents=8, circle_radius=4.)
    else:
        agents = test_case_fn(**test_case_args)
    
    env.set_agents(agents)
    init_obs = env.reset()
    env.unwrapped.test_case_index = test_case
    return init_obs, agents


def main():
    np.random.seed(0)

    Config.EVALUATE_MODE = True
    Config.SAVE_EPISODE_PLOTS = True
    Config.SHOW_EPISODE_PLOTS = True
    Config.ANIMATE_EPISODES = False
    Config.DT = 0.1
    Config.USE_STATIC_MAP = False
    Config.PLOT_CIRCLES_ALONG_TRAJ = True
    Config.RECORD_PICKLE_FILES = True
    Config.GENERATE_DATASET = True
    Config.PLT_LIMITS = [[-8, 8], [-8, 8]]

    # REWARD params
    rewardtype = 'default'
    Config.REWARD_AT_GOAL = rewCONF.getfloat(rewardtype, 'reach_goal')
    Config.REWARD_COLLISION_WITH_AGENT = rewCONF.getfloat(rewardtype, 'collision_agent')
    Config.REWARD_COLLISION_WITH_WALL = -rewCONF.getfloat(rewardtype, 'collision_wall')
    Config.REWARD_GETTING_CLOSE   = rewCONF.getfloat(rewardtype, 'close_reward')
    Config.GETTING_CLOSE_RANGE = rewCONF.getfloat(rewardtype, 'close_range')
    Config.REWARD_TIME_STEP   = rewCONF.getfloat(rewardtype, 'timestep')
    Config.REACHER = rewCONF.getboolean(rewardtype, 'reacher')

    # Data Gen params
    # num_agents_to_test = range(10,11)
    num_agents_to_test = [4]
    # num_agents_to_test = ['multi']
    num_test_cases = 5000
    policies = ['circle8']

    test_case_fn = tc.get_testcase_random
    test_case_args = {
            'policy_to_ensure': None,
            'policies': ['RVO', 'noncoop', 'static', 'random'],
            # 'policy_distr': [0.75, 0.10, 0.075, 0.075],
            'policy_distr': [1, 0, 0, 0],
            'speed_bnds': [0.5, 2.0],
            'radius_bnds': [0.2, 0.8],
            'side_length': [
                {'num_agents': [0,5], 'side_length': [2,4]}, 
                {'num_agents': [5,np.inf], 'side_length': [4,6]},
                ],
            'agents_sensors': ['other_agents_states'],
        }
    #######################################################################
    env = create_env()

    print(
        "Running {test_cases} test cases for {num_agents} for policies:"
        " {policies}".format(
            test_cases=num_test_cases,
            num_agents=num_agents_to_test,
            policies=policies,
        )
    )
    with tqdm(
        total=len(num_agents_to_test)
        * len(policies)
        * num_test_cases
    ) as pbar:
        for num_agents in num_agents_to_test:
            for policy in policies:
                env.set_plot_save_dir(
                os.path.dirname(os.path.realpath(__file__))
                + "/DATA/results/{policy}_{num_agents}_agents/figs/"
                .format(policy=policy, num_agents=num_agents)
            )
                np.random.seed(0)
                prev_agents = None
                df = pd.DataFrame()
                datasets = []
                for test_case in range(num_test_cases):
                    ##### Actually run the episode ##########
                    init_obs, _ = reset_env(
                        env,
                        test_case_fn,
                        test_case_args,
                        test_case,
                        num_agents,
                        policies,
                        policy,
                        prev_agents,
                    )
                    episode_stats, prev_agents, dataset = run_episode(env)
                    datasets.append(dataset)

                    # print(episode_stats)
                    df = store_stats(
                        df,
                        {"test_case": test_case, "policy_id": policy},
                        episode_stats,
                    )
                    logging.info(f'EPISODE {test_case}: {episode_stats}')
                    ########################################
                    pbar.update(1)

                if Config.GENERATE_DATASET:
                    file_dir = os.path.dirname(os.path.realpath(__file__))+ "/DATA/datasets/{}_{}_agent_{}".format(policy, num_agents, num_test_cases)
                    with open(file_dir + '.pkl', 'wb') as handle:
                        pickle.dump(datasets, handle, protocol=pickle.HIGHEST_PROTOCOL)       
                    print(f'Generated Dataset Length: {len(datasets)}')

                if Config.RECORD_PICKLE_FILES:
                    file_dir = os.path.dirname(os.path.realpath(__file__)) + "/DATA/results/"
                    file_dir += "{policy}_{num_agents}_agents/stats/".format(policy=policy, num_agents=num_agents)
                    os.makedirs(file_dir, exist_ok=True)
                    log_filename = file_dir + "/{}_{}_{}_stats.p".format(policy, num_agents, num_test_cases)
                    df.to_pickle(log_filename)
    return True


if __name__ == "__main__":
    logging.basicConfig(filename=os.path.dirname(os.path.realpath(__file__))
                        + "/DATA/datasets/DATASET.log", filemode='w', level=logging.DEBUG)
    logging.info('started')
    main()
    logging.info('finished')
    print("Experiment over.")