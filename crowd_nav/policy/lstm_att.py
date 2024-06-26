import torch
import torch.nn as nn
import numpy as np
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL


class ValueNetwork1(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp_dims, mlp1_dims, attention_dims, lstm_hidden_dim):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp = mlp(self_state_dim + lstm_hidden_dim, mlp_dims)
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)

        # for attention module
        self.mlp1 = mlp(13, mlp1_dims, last_relu=True)
        self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)


    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a joint state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        human_state = state[:, :, self.self_state_dim:]

        # add attention module
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
        global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
            contiguous().view(-1, self.global_state_dim)
        attention_input = torch.cat([mlp1_output, global_state], dim=1)
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)
        rerank_human_state = np.zeros(human_state.shape)
        for i in range(size[0]):
            score = scores[i, :]
            ind = torch.argsort(score)
            rerank_human_state[i, :, :] = human_state[i, ind, :]

        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        # scores_exp = torch.exp(scores) * (scores != 0).float()
        # weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)


        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        output, (hn, cn) = self.lstm(human_state, (h0, c0))
        hn = hn.squeeze(0)
        joint_state = torch.cat([self_state, hn], dim=1)
        value = self.mlp(joint_state)
        return value


class ValueNetwork2(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp_dims, lstm_hidden_dim):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.mlp1 = mlp(input_dim, mlp1_dims)
        self.mlp = mlp(self_state_dim + lstm_hidden_dim, mlp_dims)
        self.lstm = nn.LSTM(mlp1_dims[-1], lstm_hidden_dim, batch_first=True)

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a joint state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]

        state = torch.reshape(state, (-1, size[2]))
        mlp1_output = self.mlp1(state)
        mlp1_output = torch.reshape(mlp1_output, (size[0], size[1], -1))

        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        output, (hn, cn) = self.lstm(mlp1_output, (h0, c0))
        hn = hn.squeeze(0)
        joint_state = torch.cat([self_state, hn], dim=1)
        value = self.mlp(joint_state)
        return value


class LSTM_ATT(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'LSTM-ATT'
        self.with_interaction_module = None
        self.interaction_module_dims = None

    def configure(self, config):
        self.set_common_parameters(config)
        mlp_dims = [int(x) for x in config.get('lstm_att', 'mlp2_dims_lstm').split(', ')]
        mlp1_dims = [int(x) for x in config.get('lstm_att', 'mlp1_dims').split(', ')]
        global_state_dim = config.getint('lstm_att', 'global_state_dim')
        self.with_om = config.getboolean('lstm_att', 'with_om')
        with_interaction_module = config.getboolean('lstm_att', 'with_interaction_module')
        attention_dims = [int(x) for x in config.get('lstm_att', 'attention_dims').split(', ')]
        if with_interaction_module:
            mlp1_dims = [int(x) for x in config.get('lstm_att', 'mlp1_dims_lstm').split(', ')]
            self.model = ValueNetwork2(self.input_dim(), self.self_state_dim, mlp1_dims, mlp_dims, global_state_dim)
        else:
            self.model = ValueNetwork1(7, self.self_state_dim, mlp_dims, mlp1_dims, attention_dims, global_state_dim)
        self.multiagent_training = config.getboolean('lstm_att', 'multiagent_training')
        logging.info('Policy: {}LSTM-ATT {} pairwise interaction module'.format(
            'OM-' if self.with_om else '', 'w/' if with_interaction_module else 'w/o'))

    def predict(self, state):
        """
        Input state is the joint state of robot concatenated with the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """

        def dist(human):
            # sort human order by decreasing distance to the robot
            return np.linalg.norm(np.array(human.position) - np.array(state.self_state.position))

        state.human_states = sorted(state.human_states, key=dist, reverse=True)
        return super().predict(state)
