import copy

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
from collections import deque


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output, p_attn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        sizes = [self.config["in"]] + self.config["layers"]
        self.activation = F.relu
        layers_list = [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.layers = nn.ModuleList(layers_list)
        if self.config.get("out", None):
            self.predict = nn.Linear(sizes[-1], self.config["out"])

    def forward(self, x):
        if self.config["reshape"]:
            x = x.reshape(x.shape[0], -1)  # We expect a batch of vectors
        for layer in self.layers:
            x = self.activation(layer(x))
        if self.config.get("out", None):
            x = self.predict(x)
        return x


class EgoAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.features_per_head = int(self.config["feature_size"] / self.config["heads"])
        self.value_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        self.key_all = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        self.query_ego = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)
        self.attention_combine = nn.Linear(self.config["feature_size"], self.config["feature_size"], bias=False)

    def _init_weights(self, m):
        if hasattr(m, 'weight'):
            torch.nn.init.xavier_uniform_(m.weight.data)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.)

    def forward(self, ego, others, mask=None):
        batch_size = others.shape[0]
        n_entities = others.shape[1] + 1
        input_all = torch.cat((ego.view(batch_size, 1, self.config["feature_size"]), others), dim=1)
        # Dimensions: Batch, entity, head, feature_per_head
        key_all = self.key_all(input_all).view(batch_size, n_entities, self.config["heads"], self.features_per_head)
        value_all = self.value_all(input_all).view(batch_size, n_entities, self.config["heads"], self.features_per_head)
        query_ego = self.query_ego(ego).view(batch_size, 1, self.config["heads"], self.features_per_head)

        # Dimensions: Batch, head, entity, feature_per_head
        key_all = key_all.permute(0, 2, 1, 3)
        value_all = value_all.permute(0, 2, 1, 3)
        query_ego = query_ego.permute(0, 2, 1, 3)
        if mask is not None:
            mask = mask.view((batch_size, 1, 1, n_entities)).repeat((1, self.config["heads"], 1, 1))
        value, attention_matrix = attention(query_ego, key_all, value_all, mask,
                                            nn.Dropout(self.config["dropout_factor"]))
        result = (self.attention_combine(value.reshape((batch_size, self.config["feature_size"]))) + ego.squeeze(1)) / 2
        return result, attention_matrix


class DRQN(nn.Module):
    def __init__(self, input_shape, conf):
        super().__init__()
        self.conf = conf
        self.feature_num = 16
        self.input_num = 7
        ego_config = {
            'layers': [self.feature_num],
            'reshape': False,
            'in': self.input_num,
            'out': None
        }
        other_config = {
            'layers': [self.feature_num],
            'reshape': False,
            'in': self.input_num,
        }
        atten_config = {"feature_size": self.feature_num,
                        "heads": 2,
                        "dropout_factor": 0
                        }
        self.ego_embedding = MultiLayerPerceptron(ego_config)
        self.others_embedding = MultiLayerPerceptron(other_config)
        self.attention_layer = EgoAttention(atten_config)

        if self.conf.obs_shape % 7 == 0 and self.conf.attention:
            input_shape = self.feature_num
        else:
            if not self.conf.use_id:
                input_shape = self.conf.obs_shape
        if self.conf.use_rnn:
            self.fc1 = nn.Linear(input_shape, conf.drqn_hidden_dim)
            if self.conf.mix_network == "maven":
                self.rnn = nn.GRUCell(conf.drqn_hidden_dim + conf.noise_d, conf.drqn_hidden_dim)
            else:
                self.rnn = nn.GRUCell(conf.drqn_hidden_dim, conf.drqn_hidden_dim)
            self.fc2 = nn.Linear(conf.drqn_hidden_dim, conf.n_actions)
        else:
            self.fc1 = nn.Linear(input_shape, conf.drqn_hidden_dim)
            if self.conf.mix_network == "maven":
                self.fc = nn.Linear(conf.drqn_hidden_dim + conf.noise_d, conf.drqn_hidden_dim_2)
            else:
                self.fc = nn.Linear(conf.drqn_hidden_dim, conf.drqn_hidden_dim_2)
            self.fc2 = nn.Linear(conf.drqn_hidden_dim_2, conf.n_actions)

        self.init_weights()

    def forward(self, obs, hidden_state, z=torch.zeros(16)):
        if obs.shape[1] % 7 == 0 and self.conf.attention:
            obs = obs.view(obs.shape[0], -1, 7)
            ego_embedded_att, attention_matrix = self.forward_attention(obs)
            x = F.relu(self.fc1(ego_embedded_att))
        else:
            x = F.relu(self.fc1(obs))

        if self.conf.mix_network == "maven":
            z = z.to(x.device)
            if len(z.size()) < len(x.size()):
                z = z.repeat(x.size(0), 1)
            x = torch.cat([x, z], dim=-1)

        if self.conf.use_rnn:
            h_in = hidden_state.reshape(-1, self.conf.drqn_hidden_dim)
            h = self.rnn(x, h_in)
            q = self.fc2(h)
        else:
            q = self.fc2(F.relu(self.fc(x)))
            h = hidden_state
        return q, h

    def forward_attention(self, x):
        ego, others, mask = self.split_input(x)
        ego, others = self.ego_embedding(ego), self.others_embedding(others)
        return self.attention_layer(ego, others, mask)

    def split_input(self, x, mask=None):
        # Dims: batch, entities, features
        ego = x[:, 0:1, :]
        others = x[:, 1:, :]
        if mask is None:
            mask = x[:, :, 0:1] < 0.4  # others are less than 0.5
        return ego, others, mask

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class QMIXNET(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        if self.conf.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(self.conf.state_shape, self.conf.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.conf.hyper_hidden_dim,
                                                    self.conf.n_agents * self.conf.qmix_hidden_dim))
            self.hyper_w2 = nn.Sequential(nn.Linear(self.conf.state_shape, self.conf.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.conf.hyper_hidden_dim, self.conf.qmix_hidden_dim))
        else:
            self.hyper_w1 = nn.Linear(self.conf.state_shape, self.conf.n_agents * self.conf.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(self.conf.state_shape, self.conf.qmix_hidden_dim * 1)

        self.hyper_b1 = nn.Linear(self.conf.state_shape, self.conf.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(self.conf.state_shape, self.conf.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.conf.qmix_hidden_dim, 1))

    # input: (batch_size, n_agents, qmix_hidden_dim)
    # q_values: (episode_num, max_episode_len, n_agents)
    # states shape: (episode_num, max_episode_len, state_shape)
    def forward(self, q_values, states, z=torch.zeros(16)):
        # print(self.conf.state_shape)

        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.conf.n_agents)
        if self.conf.mix_network == "maven":
            states = states.reshape(-1, self.conf.state_shape - self.conf.noise_d)
            z = z.reshape(-1, self.conf.noise_d)
            states = torch.cat([states, z.to(states.device)], dim=-1)
        else:
            states = states.reshape(-1, self.conf.state_shape)

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = w1.view(-1, self.conf.n_agents, self.conf.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.conf.qmix_hidden_dim)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)
        w2 = w2.view(-1, self.conf.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(episode_num, -1, 1)

        return q_total

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class VDNNet(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

    def forward(self, q_values, states):
        return torch.sum(q_values, dim=2, keepdim=True)


class QresNet(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        if self.conf.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(self.conf.state_shape, self.conf.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.conf.hyper_hidden_dim,
                                                    self.conf.n_agents * self.conf.qmix_hidden_dim))
            self.hyper_w2 = nn.Sequential(nn.Linear(self.conf.state_shape, self.conf.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.conf.hyper_hidden_dim, self.conf.qmix_hidden_dim))
        else:
            self.hyper_w1 = nn.Linear(self.conf.state_shape, self.conf.n_agents * self.conf.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(self.conf.state_shape, self.conf.qmix_hidden_dim * 1)

        self.hyper_b1 = nn.Linear(self.conf.state_shape, self.conf.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(self.conf.state_shape, self.conf.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.conf.qmix_hidden_dim, 1))
        self.init_weights()

    def forward(self, q_values, states):
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.conf.n_agents)
        states = states.reshape(-1, self.conf.state_shape)

        w1 = self.hyper_w1(states)
        b1 = self.hyper_b1(states)
        w1 = w1.view(-1, self.conf.n_agents, self.conf.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.conf.qmix_hidden_dim)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)

        w2 = self.hyper_w2(states)
        b2 = self.hyper_b2(states)
        w2 = w2.view(-1, self.conf.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.view(episode_num, -1, 1)

        return q_total


class Policy(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = copy.deepcopy(args)
        self.affine1 = nn.Linear(args.state_shape, 128)
        self.affine2 = nn.Linear(128, args.noise_d)

    def forward(self, x):
        x = x.view(-1, self.args.state_shape)
        x = self.affine1(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


# maven
class Z_agent:
    def __init__(self, args):
        self.conf = copy.deepcopy(args)
        self.lr = args.lr
        self.noise_dim = self.conf.noise_d
        # size of state vector
        self.state_shape = self.conf.state_shape
        self.policy = Policy(self.conf)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

    def sample(self, state, mode):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.conf.device)
        probs = self.policy(state)
        m = torch.distributions.one_hot_categorical.OneHotCategorical(probs)
        action = m.sample()
        return action

    def update_returns(self, states, actions, returns, test_mode):
        if test_mode:
            return
        probs = self.policy(states)
        m = torch.distributions.one_hot_categorical.OneHotCategorical(probs)
        log_probs = m.log_prob(actions)
        self.optimizer.zero_grad()
        policy_loss = -torch.dot(log_probs, returns)
        policy_loss.backward()
        self.optimizer.step()

    def cuda(self):
        self.policy.to(self.conf.device)


# Max entropy Z agent
class EZ_agent:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.lr = args.lr
        self.noise_dim = self.args.noise_d
        # size of state vector
        self.state_shape = self.args.state_shape
        self.policy = Policy(args)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        # Scaling factor for entropy, would roughly be similar to MI scaling
        self.entropy_scaling = args.entropy_scaling
        self.uniform_distrib = torch.distributions.one_hot_categorical.OneHotCategorical(
            torch.tensor([1 / self.args.noise_d for _ in range(self.args.noise_d)]))

        self.buffer = deque(maxlen=self.args.bandit_buffer)

    def sample(self, state, test_mode=False):
        # During testing we just sample uniformly
        if test_mode:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.args.device)
            probs = self.policy(state)
            max_indices = torch.argmax(probs, dim=1)
            num_classes = probs.shape[1]
            return F.one_hot(max_indices, num_classes=num_classes).cpu()
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.args.device)
            probs = self.policy(state)
            m = torch.distributions.one_hot_categorical.OneHotCategorical(probs)
            action = m.sample().cpu()
            # return self.uniform_distrib.sample()
            return action

    def update(self, mini_batch):
        states = mini_batch['s'][:, 0]
        actions = mini_batch['z'][:, 0]
        r = mini_batch['r'].squeeze()
        return_m = self.args.return_m.repeat(r.size()[0], 1)
        retrun = return_m * r
        returns = torch.sum(retrun, dim=1)
        self.update_returns(states, actions, returns, False, 1)

    def update_returns(self, states, actions, returns, test_mode, t):
        if test_mode:
            return

        for s, a, r in zip(states, actions, returns):
            self.buffer.append((s, a, r))

        for _ in range(self.args.bandit_iters):
            idxs = np.random.randint(0, len(self.buffer), size=self.args.bandit_batch)
            batch_elems = [self.buffer[i] for i in idxs]
            states_ = torch.stack([x[0] for x in batch_elems]).to(self.args.device)
            actions_ = torch.stack([x[1] for x in batch_elems]).to(self.args.device)
            returns_ = torch.stack([x[2] for x in batch_elems]).to(self.args.device)

            probs = self.policy(states_)
            m = torch.distributions.one_hot_categorical.OneHotCategorical(probs)
            log_probs = m.log_prob(actions_.to(probs.device))
            self.optimizer.zero_grad()
            policy_loss = -torch.dot(log_probs, returns_.clone().detach()) + self.entropy_scaling * log_probs.sum()
            policy_loss.backward()
            self.optimizer.step()

    def cuda(self):
        self.policy.to(self.args.device)

    def save_model(self, path):
        torch.save(self.policy.state_dict(), "{}/ez_bandit_policy.th".format(path))
