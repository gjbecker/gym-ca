import numpy as np
from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy

class RandomPolicy(InternalPolicy):
    """ Random Agents simply drive at random speeds and in random directions, ignoring other agents. """
    def __init__(self):
        InternalPolicy.__init__(self, str="RandomPolicy")

    def find_next_action(self, obs, agents, i):
        """ Go at random speed [0,2), apply a random change in heading relative to ego_head to stay within [-2*pi,2*pi]

        Args:
            obs (dict): ignored
            agents (list): of Agent objects
            i (int): this agent's index in that list

        Returns:
            np array of shape (2,)... [spd, delta_heading]

        """
        heading = agents[i].heading_ego_frame
        if heading >= 0:
            delta_head = np.random.uniform(-2*np.pi, (2*np.pi)-heading)
        else:
           delta_head = np.random.uniform((-2*np.pi)-heading, 2*np.pi)

        action = np.array([np.random.uniform(0, 2), delta_head])
        return action