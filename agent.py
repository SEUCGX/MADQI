import numpy as np
import torch
from policy import QMIX, VDN, resQ
class Agents:
    def __init__(self, conf):
        self.conf = conf
        self.device = conf.device
        self.n_actions = conf.n_actions
        self.n_agents = conf.n_agents
        self.state_shape = conf.state_shape
        self.obs_shape = conf.obs_shape
        self.episode_limit = conf.episode_limit
        if conf.mix_network == "VDN":
            self.policy = VDN(conf)
        elif conf.mix_network == "QMIX" or conf.mix_network == "WQMIX":
            self.policy = QMIX(conf)
        elif conf.mix_network == "resQ":
            self.policy = resQ(conf)
        elif conf.mix_network == "maven":
            self.policy = QMIX(conf)
        self.save_times = 0
        self.human_data = {"obs": [], "human_action": [], "agent_action": []}  # for record
        self.human_buffer = {"obs": [], "human_action": [], "agent_action": []}
        self.human_takeover = conf.human_takeover
        self.mode = "train"
        self.z = torch.zeros(1, 16).to(self.conf.device)
        print("Agents inited!")

    def choose_action(self, obs, last_action, agent_num, availible_actions, epsilon, evaluate=False):
        # print(availible_actions)
        availible_actions_idx = np.nonzero(availible_actions)[0]
        q_value = self.get_q_value(obs, last_action, agent_num, availible_actions)
        if np.random.uniform() < epsilon and not evaluate and self.conf.mix_network != "maven":
            action = np.random.choice(availible_actions_idx)
        else:
            action = int(torch.argmax(q_value).cpu().numpy())
        return action

    def get_action_probability(self, obs, last_action, agent_num, availible_actions, evaluate_idx=-1):
        q_value = self.get_q_value(obs, last_action, agent_num, availible_actions)
        if evaluate_idx == -1:
            return q_value
        else:
            return q_value.softmax(dim=1)[:, evaluate_idx]

    def get_q_value(self, obs, last_action, agent_num, availible_actions):
        inputs = obs.copy()
        agents_id = np.zeros(self.n_agents)
        agents_id[agent_num] = 1.

        if self.conf.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.conf.reuse_network and self.conf.use_id:
            inputs = np.hstack((inputs, agents_id))
        hidden_state = self.policy.eval_hidden[:, agent_num, :].to(self.device)

        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(self.device)  # (42,) -> (1,42)
        availible_actions = torch.tensor(availible_actions, dtype=torch.float32).unsqueeze(0).to(self.device)

        # get q value
        if not hasattr(self, 'z'):
            setattr(self, 'z', torch.zeros(1, 16).to(self.conf.device))
        q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_drqn_net(inputs, hidden_state, self.z)
        # choose action form q value
        q_value[availible_actions == 0.0] = -float("inf")

        return q_value

    def get_ig_action_probability(self, obs, last_action, agent_num, availible_actions, evaluate_idx=-1):
        q_value = self.get_ig_q_value(obs, last_action, agent_num, availible_actions)
        if evaluate_idx == -1:
            return q_value
        else:
            return q_value.softmax(dim=1)[:, evaluate_idx]

    def get_ig_q_value(self, inputs, last_action, agent_num, availible_actions):
        hidden_state = self.policy.eval_hidden[:, agent_num, :].to(self.device)
        q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_drqn_net(inputs, hidden_state)
        return q_value

    def _get_max_episode_len(self, batch):
        terminated = batch["terminated"]
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        if len(self.human_buffer["obs"]) > 10:
            self.policy.learn(batch, max_episode_len, train_step, epsilon, self.human_buffer)
        else:
            self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.conf.save_frequency == 0:
            self.policy.save_model(train_step)
