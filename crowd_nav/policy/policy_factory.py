from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.lstm_att import LSTM_ATT
from crowd_nav.policy.lstm_mha import LSTM_MHA
from crowd_nav.policy.mha import MHA
from crowd_nav.policy.sarl import SARL

policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['lstm_att'] = LSTM_ATT
policy_factory['lstm_mha'] = LSTM_MHA
policy_factory['mha'] = MHA
