import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL


class ValueNetwork(nn.Module):
    def __init__(self, human_num, input_dim, self_state_dim, embedding_dim):
        super().__init__()
        self.self_state_dim = self_state_dim
        human_dim = input_dim - self_state_dim
        self.crowd_embedding = nn.Sequential(
                    nn.Linear(human_num * human_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, embedding_dim)
                    )
        self.robot_embedding = nn.Sequential(
                    nn.Linear(self_state_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, embedding_dim)
                    )

        self.encoder =  nn.Sequential(
                    nn.Linear(2 * embedding_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1)
                    )
    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        human_state = state[:, :, self.self_state_dim:]
        human_state = torch.reshape(human_state, (size[0], -1))
        crowd_emb= self.crowd_embedding(human_state)
        robot_emb = self.robot_embedding(self_state)
        feature = torch.cat([robot_emb,crowd_emb], dim=1)
        value = self.encoder(feature)

        return value


class SimpleMLP(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'SimpleMLP'

    def configure(self, config):
        self.set_common_parameters(config)
        human_num = config.getint('simple_mlp', 'human_num')
        embedding_dim = config.getint('simple_mlp', 'embedding_dim')
        input_dim = 13
        self_state_dim = 6

        self.model = ValueNetwork(human_num, input_dim, self_state_dim, embedding_dim)
        self.multiagent_training = config.getboolean('simple_mlp', 'multiagent_training')

    # def get_attention_weights(self):
    #     return self.model.attention_weights
