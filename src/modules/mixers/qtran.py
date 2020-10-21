import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QTranBase(nn.Module):
    def __init__(self, args):
        super(QTranBase, self).__init__()

        self.args = args

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.arch = self.args.qtran_arch # QTran architecture

        self.embed_dim = args.mixing_embed_dim

        # Q(s,u)
        if self.arch == "coma_critic":
            # Q takes [state, u] as input
            q_input_size = self.state_dim + (self.n_agents * self.n_actions)
        elif self.arch == "qtran_paper":
            # Q takes [state, agent_action_observation_encodings]
            q_input_size = self.state_dim + self.args.rnn_hidden_dim + self.n_actions
        else:
            raise Exception("{} is not a valid QTran architecture".format(self.arch))

        if self.args.network_size == "small":
            self.Q = nn.Sequential(nn.Linear(q_input_size, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))

            # V(s)
            self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))
            ae_input = self.args.rnn_hidden_dim + self.n_actions
            self.action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input),
                                                 nn.ReLU(),
                                                 nn.Linear(ae_input, ae_input))
        elif self.args.network_size == "big":
            self.Q = nn.Sequential(nn.Linear(q_input_size, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))
            # V(s)
            self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))
            ae_input = self.args.rnn_hidden_dim + self.n_actions
            self.action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input),
                                                 nn.ReLU(),
                                                 nn.Linear(ae_input, ae_input))
        else:
            assert False

    def forward(self, batch, hidden_states, actions=None):
        bs = batch.batch_size
        ts = batch.max_seq_length

        states = batch["state"].reshape(bs * ts, self.state_dim)

        if self.arch == "coma_critic":
            if actions is None:
                # Use the actions taken by the agents
                actions = batch["actions_onehot"].reshape(bs * ts, self.n_agents * self.n_actions)
            else:
                # It will arrive as (bs, ts, agents, actions), we need to reshape it
                actions = actions.reshape(bs * ts, self.n_agents * self.n_actions)
            inputs = th.cat([states, actions], dim=1)
        elif self.arch == "qtran_paper":
            if actions is None:
                # Use the actions taken by the agents
                actions = batch["actions_onehot"].reshape(bs * ts, self.n_agents, self.n_actions)
            else:
                # It will arrive as (bs, ts, agents, actions), we need to reshape it
                actions = actions.reshape(bs * ts, self.n_agents, self.n_actions)

            hidden_states = hidden_states.reshape(bs * ts, self.n_agents, -1)
            agent_state_action_input = th.cat([hidden_states, actions], dim=2)
            agent_state_action_encoding = self.action_encoding(agent_state_action_input.reshape(bs * ts * self.n_agents, -1)).reshape(bs * ts, self.n_agents, -1)
            agent_state_action_encoding = agent_state_action_encoding.sum(dim=1) # Sum across agents

            inputs = th.cat([states, agent_state_action_encoding], dim=1)

        q_outputs = self.Q(inputs)

        states = batch["state"].reshape(bs * ts, self.state_dim)
        v_outputs = self.V(states)

        return q_outputs, v_outputs


class QTranAlt(nn.Module):
    def __init__(self, args):
        super(QTranAlt, self).__init__()

        self.args = args

        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        # Q(s,-,u-i)
        # Q takes [state, u-i, i] as input
        q_input_size = self.state_dim + (self.n_agents * self.n_actions) + self.n_agents

        if self.args.network_size == "small":
            self.Q = nn.Sequential(nn.Linear(q_input_size, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.n_actions))

            # V(s)
            self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))
        elif self.args.network_size == "big":
             # Adding another layer
             self.Q = nn.Sequential(nn.Linear(q_input_size, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.n_actions))
            # V(s)
             self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))
        else:
            assert False

    def forward(self, batch, masked_actions=None):
        bs = batch.batch_size
        ts = batch.max_seq_length
        # Repeat each state n_agents times
        repeated_states = batch["state"].repeat(1, 1, self.n_agents).view(-1, self.state_dim)

        if masked_actions is None:
            actions = batch["actions_onehot"].repeat(1, 1, self.n_agents, 1)
            agent_mask = (1 - th.eye(self.n_agents, device=batch.device))
            agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions)#.view(self.n_agents, -1)
            masked_actions = actions * agent_mask.unsqueeze(0).unsqueeze(0)
            masked_actions = masked_actions.view(-1, self.n_agents * self.n_actions)

        agent_ids = th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).repeat(bs, ts, 1, 1).view(-1, self.n_agents)

        inputs = th.cat([repeated_states, masked_actions, agent_ids], dim=1)

        q_outputs = self.Q(inputs)

        states = batch["state"].repeat(1,1,self.n_agents).view(-1, self.state_dim)
        v_outputs = self.V(states)

        return q_outputs, v_outputs


