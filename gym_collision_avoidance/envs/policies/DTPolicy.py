import numpy as np
import os, sys
import operator
import torch
from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy
from gym_collision_avoidance.envs import util
from gym_collision_avoidance.envs import Config

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../dt-ca')))
print(sys.path)
try:
    from decision_transformer.models.decision_transformer_linear import DecisionTransformer
    import GA3C_config as DTconfig
except:
    sys.exit('ERROR: dt-ca not found!')

class DTPolicy(InternalPolicy):
    """ Pre-trained policy from `Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning <https://arxiv.org/pdf/1805.01956.pdf>`_

    By default, loads a pre-trained LSTM network (GA3C-CADRL-10-LSTM from the paper). There are 11 discrete actions with max heading angle change of $\pm \pi/6$.

    """
    def __init__(self):
        InternalPolicy.__init__(self, str="DT")

    def initialize_network(self, state_dim, act_dim, model_path, device):
        self.device = device
        self.model = DecisionTransformer(
                state_dim,
                act_dim,
                max_length=DTconfig.k,
                max_ep_len=DTconfig.max_ep_len,
                hidden_size=DTconfig.embed_dim,
                n_layer=DTconfig.n_layer,
                n_head=DTconfig.n_head,
                n_inner=4*DTconfig.embed_dim,
                activation_function=DTconfig.activation_fn,
                n_positions=1024,
                resid_pdrop=DTconfig.dropout,
                attn_pdrop=DTconfig.dropout,
            )
        self.model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        self.model.eval()
        self.model.to(device)

    def find_next_action(self, states, actions, rewards, target_return, timesteps):
        """ Using only the dictionary obs, convert this to the vector needed for the GA3C-CADRL network, query the network, adjust the actions for this env.

        Args:
            obs (dict): this :class:`~gym_collision_avoidance.envs.agent.Agent` 's observation vector
            agents (list): [unused] of :class:`~gym_collision_avoidance.envs.agent.Agent` objects
            i (int): [unused] index of agents list corresponding to this agent
        
        Returns:
            [spd, heading change] command

        """

        action = self.model.get_action(
            states,
            actions.to(device=self.device, dtype=torch.float32),
            rewards.to(device=self.device, dtype=torch.float32),
            target_return.to(device=self.device, dtype=torch.float32),
            timesteps.to(device=self.device, dtype=torch.long),
        )
        return action

if __name__ == '__main__':
    policy = DTPolicy()
    policy.initialize_network(26, 29, '/work/flemingc/gjbecker/dt-ca/models/disc_GA3C-4-20240430-0129.pt', 'cpu')