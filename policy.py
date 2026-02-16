import copy

import numpy as np
import torch
import os

from torch import nn

from NN import DRQN, QMIXNET, VDNNet, QresNet, EZ_agent
from base_function import to_tensor


class MixNetwork:
    def __init__(self, conf):

        self.minibatch_size = 50
        self.weight = conf.w
        self.ex_w = conf.pw
        self.ag_w = conf.nw
        self.train_human = True
        self.optimizer = None
        self.conf = copy.deepcopy(conf)
        if conf.mix_network == "maven":
            self.conf.state_shape += self.conf.noise_d
        self.device = self.conf.device
        self.n_actions = self.conf.n_actions
        self.n_agents = self.conf.n_agents
        self.state_shape = self.conf.state_shape
        self.obs_shape = self.conf.obs_shape
        self.model_dir = self.conf.model_dir
        # print(self.device, self.n_actions, self.n_agents, self.state_shape, self.obs_shape, input_shape)

        # DRQN
        input_shape = self.obs_shape
        if self.conf.last_action:
            input_shape += self.n_actions
        if self.conf.reuse_network and self.conf.use_id:
            input_shape += self.n_agents

        # NET
        self.eval_drqn_net = DRQN(input_shape, self.conf).to(self.device)
        self.target_drqn_net = DRQN(input_shape, self.conf).to(self.device)

        # mix_network
        self.eval_mix_net = None
        self.target_mix_net = None
        self.Q_jt_target = None
        self.Q_jt = None

        self.eval_hidden = None
        self.target_hidden = None
        self.eval_parameters = None
        discrim_input = conf.state_shape + self.n_actions * self.n_agents
        self.discrim = Discrim(discrim_input, self.conf.noise_d, self.conf).to(self.conf.device)
        self.discrim_loss = torch.nn.CrossEntropyLoss(reduction="none")

        print("init mix nets finished!")

    def learn(self, batch, max_episode_len, train_step, epsilon=None, human_data=None):
        """
        batch: train data, obs: (batch_size, episode_limit, n_agents, obs_shape),(64, -53- ,3,42)
        max_episode_len: max episode length
        train_step: step record for updating target network parameters
        """
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)

        s, s_, z, u, r, avail_u, avail_u_, terminated = batch['s'], batch['s_'], batch['z'], batch['u'], batch['r'], \
                                                        batch['avail_u'], batch['avail_u_'], batch['terminated']
        mask = 1 - batch['padded'].float()

        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        s = s.to(self.device)
        u = u.to(self.device)
        r = r.to(self.device)
        s_ = s_.to(self.device)
        z = z.to(self.device)
        terminated = terminated.to(self.device)
        mask = mask.to(self.device)
        #
        u_onehot = batch['u_onehot'].to(self.device)
        actions = u_onehot.view(u_onehot.shape[0], u_onehot.shape[1], -1)
        s_and_softa_reshaped = torch.cat((s_[:, 0, :], actions[:, 0, :]), dim=-1)
        discrim_prediction = self.discrim(s_and_softa_reshaped)
        discrim_target = z[:, 0].long().detach().max(dim=1)[1].unsqueeze(1).repeat(1, 1).reshape(-1)
        masked_discrim_loss = self.discrim_loss(discrim_prediction, discrim_target)
        averaged_discrim_loss = masked_discrim_loss.mean()

        max_q = torch.max(q_evals, dim=3)[0]
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        w_evals = torch.where(max_q == q_evals, torch.tensor(1).to(max_q.device), torch.tensor(0).to(max_q.device)).sum(
            dim=-1, keepdim=True)
        w_evals = (w_evals > 1).int()
        q_targets[avail_u_ == 0.0] = -9999999
        q_targets = q_targets.max(dim=3)[0]
        if self.conf.mix_network == "maven":
            q_total_eval = self.eval_mix_net(q_evals, s, z)
            q_total_target = self.target_mix_net(q_targets, s_, z)
        else:
            q_total_eval = self.eval_mix_net(q_evals, s)
            q_total_target = self.target_mix_net(q_targets, s_)
        if self.conf.mix_network in ["WQMIX", "QMIX", "VDN"]:
            targets = r + self.conf.gamma * q_total_target * (1 - terminated)
            td_error = (q_total_eval - targets.detach())
            w_tensor = torch.where(td_error < 0, torch.tensor(1.0).to(td_error.device),
                                   torch.tensor(self.conf.wqmix_w).to(td_error.device)).to(td_error.device)
            if self.conf.mix_network == "WQMIX":
                mask_td_error = w_tensor * mask * td_error
            else:
                mask_td_error = mask * td_error
            loss = (mask_td_error ** 2).sum() / mask.sum()
        elif self.conf.mix_network == "maven":
            targets = r + self.conf.gamma * q_total_target * (1 - terminated)
            td_error = (q_total_eval - targets.detach())

            mask_td_error = mask * td_error
            loss = (mask_td_error ** 2).sum() / mask.sum() + self.conf.mi * averaged_discrim_loss

        if human_data is not None and self.train_human:
            human_loss = 1 * self.human_update(human_data)
        else:
            human_loss = 0

        loss += human_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.conf.grad_norm_clip)
        self.optimizer.step()
        if train_step > 0 and train_step % self.conf.update_target_params == 0:
            self.target_drqn_net.load_state_dict(self.eval_drqn_net.state_dict())
            self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_ = self._get_inputs(batch, transition_idx)
            inputs = inputs.to(self.device)  # [batch_size*n_agents, obs_shape+n_agents+n_actions]
            inputs_ = inputs_.to(self.device)

            self.eval_hidden = self.eval_hidden.to(self.device)
            self.target_hidden = self.target_hidden.to(self.device)
            z = batch["z"][:, transition_idx].squeeze().repeat(self.n_agents, 1)
            q_eval, self.eval_hidden = self.eval_drqn_net(inputs, self.eval_hidden, z)  # (n_agents, n_actions)
            q_target, self.target_hidden = self.target_drqn_net(inputs_, self.target_hidden, z)
            # 训练时固定z
            q_eval = q_eval.view(episode_num, self.n_agents, -1)  # (batch_size, n_agents, n_actions)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)

        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def _get_inputs(self, batch, transition_idx):
        o, o_, u_onehot = batch['o'][:, transition_idx], batch['o_'][:, transition_idx], batch['u_onehot'][:]
        episode_num = o.shape[0]  # batch_size
        inputs, inputs_ = [], []
        inputs.append(o)
        inputs_.append(o_)
        if self.conf.last_action:
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_.append(u_onehot[:, transition_idx])

        if self.conf.reuse_network and self.conf.use_id:
            inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        inputs = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs], dim=1)
        inputs_ = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs_], dim=1)

        return inputs, inputs_

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.conf.drqn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.conf.drqn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.conf.save_frequency)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        print("save model: {} epoch.".format(num))
        torch.save(self.eval_drqn_net.state_dict(), self.model_dir + '/' + num + '_drqn_net_params.pkl')
        torch.save(self.eval_mix_net.state_dict(), self.model_dir + '/' + num + '_qmix_net_params.pkl')

    def human_update(self, human_data):
        if self.conf.Monotonically_decreasing:
            self.ex_w = max(self.conf.de_f * self.ex_w, 0.1)
            self.ag_w = max(self.conf.de_f * self.ag_w, 0.0)
        if len(human_data["obs"]) < self.minibatch_size:
            return 0
        s = to_tensor(np.array(human_data["obs"]))
        ae = np.array(human_data["human_action"])
        aa = np.array(human_data["agent_action"])
        b_inds = np.arange(s.size(0))
        np.random.shuffle(b_inds)
        # minibatch_size = s.size(0) // self.human_minibatch
        start = 0
        end = start + self.minibatch_size
        mb_inds = b_inds[start:end]

        eval_hidden = torch.zeros((self.minibatch_size, 1, self.conf.drqn_hidden_dim)).to(self.device)

        q_b, _ = self.eval_drqn_net(s[mb_inds].to(self.device), eval_hidden)
        ae_b = ae[mb_inds]
        aa_b = aa[mb_inds]
        loss = 0
        for j in range(self.minibatch_size):
            max_value = torch.tensor(float("-inf")).to(self.device)
            for a in range(self.n_actions):
                max_value = torch.max(q_b[j, a] + self.loss_le(ae_b[j], a), max_value)
            min_value = torch.tensor(float("inf")).to(self.device)
            for a in range(self.n_actions):
                min_value = torch.min(q_b[j, a] + self.loss_la(aa_b[j], a), min_value)
            loss += (self.weight * (max_value - q_b[j, ae_b[j]]) + self.weight * (q_b[j, aa_b[j]] - min_value))
        loss = loss / self.minibatch_size
        # self.weight = 0.99999 * self.weight
        return loss

    def loss_le(self, ae, a):  # Boundary function learned from experts
        return 0.0 if ae == a else self.ex_w

    def loss_la(self, ae, a):  # Rejected boundary function
        return 0.0 if ae == a else -self.ag_w


class QMIX(MixNetwork):
    def __init__(self, conf):
        super().__init__(conf)
        # mix_network
        self.eval_mix_net = QMIXNET(self.conf).to(self.device)
        self.target_mix_net = QMIXNET(self.conf).to(self.device)

        if self.conf.load_model:
            if os.path.exists(self.model_dir + '/' + self.conf.load_num + '_drqn_net_params.pkl'):
                drqn_path = self.model_dir + '/' + self.conf.load_num + '_drqn_net_params.pkl'
                qmix_path = self.model_dir + '/' + self.conf.load_num + '_qmix_net_params.pkl'
                map_location = 'cuda:0' if self.conf.cuda else 'cpu'
                self.eval_drqn_net.load_state_dict(torch.load(drqn_path, map_location=map_location))
                self.eval_mix_net.load_state_dict(torch.load(qmix_path, map_location=map_location))
                print("successfully load models")
            else:
                raise Exception("No model!")

        # copy eval net params to target net
        self.target_drqn_net.load_state_dict(self.eval_drqn_net.state_dict())
        self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())

        self.eval_parameters = list(self.eval_mix_net.parameters()) + list(self.eval_drqn_net.parameters())
        if self.conf.mix_network == "maven":
            self.eval_parameters += list(self.discrim.parameters())
        if self.conf.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=self.conf.learning_rate)
        elif self.conf.optimizer == "ADAM":
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=self.conf.learning_rate)

        if self.conf.mix_network == "maven":
            self.noise_net = EZ_agent(conf)
            self.noise_net.cuda()


class resQ(QMIX):
    def __init__(self, conf):
        super().__init__(conf)
        self.Qr = QresNet(self.conf).to(self.device)
        self.Qr_target = QresNet(self.conf).to(self.device)
        self.Q_jt = QresNet(self.conf).to(self.device)
        self.Q_jt_target = QresNet(self.conf).to(self.device)


class VDN(MixNetwork):
    def __init__(self, conf):
        super().__init__(conf)
        # mix_network
        self.eval_mix_net = VDNNet(self.conf).to(self.device)
        self.target_mix_net = VDNNet(self.conf).to(self.device)

        if self.conf.load_model:
            if os.path.exists(self.model_dir + '/' + self.conf.load_num + '_drqn_net_params.pkl'):
                drqn_path = self.model_dir + '/' + self.conf.load_num + '_drqn_net_params.pkl'
                mix_path = self.model_dir + '/' + self.conf.load_num + '_qmix_net_params.pkl'
                map_location = 'cuda:0' if self.conf.cuda else 'cpu'
                self.eval_drqn_net.load_state_dict(torch.load(drqn_path, map_location=map_location))
                self.eval_mix_net.load_state_dict(torch.load(mix_path, map_location=map_location))
                print("successfully load models")
            else:
                raise Exception("No model!")

        # copy eval net params to target net
        self.target_drqn_net.load_state_dict(self.eval_drqn_net.state_dict())
        self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())

        self.eval_parameters = list(self.eval_mix_net.parameters()) + list(self.eval_drqn_net.parameters())
        if self.conf.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=self.conf.learning_rate)
        elif self.conf.optimizer == "ADAM":
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=self.conf.learning_rate)


class Discrim(nn.Module):
    def __init__(self, input_size, output_size, conf):
        super().__init__()
        self.conf = conf
        layers = [torch.nn.Linear(input_size, self.conf.discrim_size), torch.nn.ReLU()]
        for _ in range(self.conf.discrim_layers - 1):
            layers.append(torch.nn.Linear(self.conf.discrim_size, self.conf.discrim_size))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(self.conf.discrim_size, output_size))
        self.model = torch.nn.Sequential(*layers)
        self.discrim_input = input_size

    def forward(self, x):
        x = x.reshape(-1, self.discrim_input)
        return self.model(x)
