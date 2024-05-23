import numpy as np
import os, sys
import operator
import torch
from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy
from gym_collision_avoidance.envs import util
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs.policies.GA3C_CADRL.network import Actions, Actions_Plus

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../dt-ca')))
try:
    from decision_transformer.models.decision_transformer import DecisionTransformer
    import DT_GA3C_config as DTconfig
except:
    sys.exit('ERROR: dt-ca not found!')

class DTPolicy(InternalPolicy):
    """ Pre-trained policy from `Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning <https://arxiv.org/pdf/1805.01956.pdf>`_

    By default, loads a pre-trained LSTM network (GA3C-CADRL-10-LSTM from the paper). There are 11 discrete actions with max heading angle change of $\pm \pi/6$.

    """
    def __init__(self):
        InternalPolicy.__init__(self, str="DT")

    def initialize_model(self, state_dim, act_dim, model_path, device='cpu'):
        self.device = device
        self.state_dim = state_dim
        self.act_dim = act_dim
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
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)), strict=False)
        self.model.eval()
        if device == 'cuda':
            self.model.cuda()

    def initialize_history(self, init_state, target_return):
        self.states = torch.from_numpy(init_state).reshape(1, self.state_dim).to(device=self.device, dtype=torch.float32)
        self.actions = torch.zeros((0, self.act_dim), device=self.device, dtype=torch.float32)
        self.rewards = torch.zeros(0, device=self.device, dtype=torch.float32)
        self.target_return = torch.tensor(target_return, device=self.device, dtype=torch.float32).reshape(1, 1)
        self.timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)

        self.t = 0

    def model_step(self, state, reward,):
        self.t += 1
        cur_state = torch.from_numpy(state).to(device=self.device).reshape(1, self.state_dim)
        self.states = torch.cat([self.states, cur_state], dim=0)
        self.actions[-1] = self.action
        self.rewards[-1] = reward
        pred_return = self.target_return[0,-1] - (reward)
        self.target_return = torch.cat(
            [self.target_return, pred_return.reshape(1, 1)], dim=1)
        self.timesteps = torch.cat(
            [self.timesteps,
             torch.ones((1, 1), device=self.device, dtype=torch.long) * (self.t)], dim=1)
        

    def find_next_action(self, dict_obs, agents, agent_idx):
        """ Using only the dictionary obs, convert this to the vector needed for the GA3C-CADRL network, query the network, adjust the actions for this env.

        Args:
            None
        
        Returns:
            [spd, heading change] command

        """
        # Pad actions and reward for next step
        self.actions = torch.cat([self.actions, torch.zeros((1, self.act_dim), device=self.device)], dim=0)
        self.rewards = torch.cat([self.rewards, torch.zeros(1, device=self.device)])

        self.action = self.model.get_action(
            self.states.to(device=self.device, dtype=torch.float32),
            self.actions.to(device=self.device, dtype=torch.float32),
            self.rewards.to(device=self.device, dtype=torch.float32),
            self.target_return.to(device=self.device, dtype=torch.float32),
            self.timesteps.to(device=self.device, dtype=torch.long),
        )

        action = self.action.detach().cpu().numpy()
        if len(action) == 2:
            pass
        elif len(action) == 11:
            act_idx = np.argmax(action)
            action = Actions().actions[act_idx]
            action[0]*=self.states[-1][3]
        elif len(action) == 29:
            act_idx = np.argmax(action)
            action = Actions_Plus().actions[act_idx]
            action[0]*=self.states[-1][3]

        return action

if __name__ == '__main__':
    policy = DTPolicy()
    policy.initialize_model(26, 29, '/work/flemingc/gjbecker/dt-ca/models/disc_GA3C-4-20240430-0129.pt', 'cpu')