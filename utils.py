import imageio
import numpy as np
import copy
import threading
import scipy.io as sio
import torch
from scipy.io import loadmat
import random
from base_function import IO_human, action_dec, rule_dec, GT_model, to_tensor


class RolloutWorker:
    def __init__(self, env, agents, conf):
        super().__init__()
        self.conf = conf
        self.agents = agents
        self.env = env
        self.episode_limit = conf.episode_limit
        self.n_actions = conf.n_actions
        self.n_agents = conf.n_agents
        self.state_shape = conf.state_shape
        self.obs_shape = conf.obs_shape

        self.start_epsilon = np.ones(self.conf.n_agents) * conf.start_epsilon
        self.anneal_epsilon = np.ones(self.conf.n_agents) * conf.anneal_epsilon
        self.end_epsilon = np.ones(self.conf.n_agents) * conf.end_epsilon
        print('Rollout Worker inited!')

    def generate_episode(self, episode_num=None, evaluate=False, render=False, save_gif=False, human_control=False,
                         human_id=0):
        o, u, r, s, z, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], [], []
        if evaluate:
            next_obs, information = self.env.reset(episode_num)
        else:
            next_obs, information = self.env.reset()
        terminated = False
        episode_reward = 0
        last_action = np.zeros((self.conf.n_agents, self.conf.n_actions))
        self.agents.policy.init_hidden(1)
        epsilon = np.zeros(self.conf.n_agents) if evaluate else self.start_epsilon
        step = 0
        speeds = [np.mean(np.array(information["speed"]))]
        if self.conf.mix_network == "maven":
            if not evaluate:
                self.agents.z = self.agents.policy.noise_net.sample(information["state"].flatten(), evaluate)
            else:
                self.agents.z = self.agents.policy.noise_net.sample(information["state"].flatten(), evaluate)
        while not terminated:
            obs = self.obs_trans(next_obs)
            state = information["state"].flatten()
            actions, avail_actions, actions_onehot = [], [], [],
            for agent_id in range(self.n_agents):
                avail_action = self.get_avail_agent_actions(next_obs, agent_id)
                ###
                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, avail_action,
                                                   epsilon[agent_id], evaluate)
                if self.conf.use_rule:
                    action = rule_dec(action, agent_id, obs[agent_id], information)
                action_onehot = np.zeros(self.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            if render or self.conf.real_human_control:
                self.env.render()
            if human_control and self.agents.human_takeover:
                if human_id == 0:
                    human_actions = action_dec(next_obs, information) # Virtual experts to make the program run normally
                elif human_id == 1:
                    virtual_r, virtual_info = self.env.virtual_step(tuple(actions))
                    if 1 or virtual_info["min_distance"] < 10 or any(virtual_info['crashed']) or self.to_close(
                            np.array(virtual_info["ttc"])):
                        human_actions = GT_model(self.env, next_obs, information)
                    else:
                        human_actions = actions
                elif human_id == 2:  # special
                    human_actions = (0, 2)
                else:
                    human_actions = actions

                for agent_id in range(self.n_agents):
                    if self.conf.real_human_control:
                        human_action = IO_human(actions[agent_id], next_obs[agent_id])
                    else:
                        human_action = human_actions[agent_id]
                    if human_action is None:
                        file_road = "human/human_data_" + str(self.agents.save_times) + ".mat"
                        print(file_road)
                        sio.savemat(file_road, self.agents.human_data)
                        self.agents.save_times = self.agents.save_times + 1
                        self.agents.human_data = {"obs": [],
                                                  "human_action": [],
                                                  "agent_action": []}
                        self.agents.human_takeover = False
                    elif human_action == 3:
                        print("No action！")
                    else:
                        if (human_action - actions[agent_id]) != 0 and (human_id != 3 or random.random() < 0.3):
                            if self.conf.reuse_network and self.conf.use_id:
                                id_onehot = np.zeros(self.conf.n_agents)
                                id_onehot[agent_id] = 1
                                inputs = np.hstack((obs[agent_id], id_onehot))
                            else:
                                inputs = obs[agent_id]
                            if len(self.agents.human_data["human_action"]) < 5000:  # only 5000
                                self.agents.human_data["human_action"].append(int(human_action))
                                self.agents.human_data["obs"].append(inputs)
                                self.agents.human_data["agent_action"].append(int(actions[agent_id]))
                                self.agents.human_buffer["human_action"].append(int(human_action))
                                self.agents.human_buffer["obs"].append(inputs)
                                self.agents.human_buffer["agent_action"].append(int(actions[agent_id]))
                            actions[agent_id] = human_action

            next_obs, reward, terminated, _, information = self.env.step(tuple(actions))
            speeds.append(np.mean(np.array(information["speed"])))
            if self.conf.Human_reward:
                if self.conf.perfect:
                    human_actions = action_dec(next_obs, information)
                else:
                    human_model = torch.load('expert_model.pth', map_location=torch.device("cpu"))
                    human_actions = [torch.argmax(human_model(to_tensor(o).view(1, -1), torch.randn(1, 128))[0]) for o
                                     in next_obs]
                human_inspire = -0.25 * (abs(np.array(human_actions) - np.array(actions))).mean()
                reward = reward + human_inspire
            step += 1
            if step >= self.episode_limit:
                terminated = True
            o.append(obs)
            s.append(state)
            z.append(self.agents.z[0].clone().cpu().numpy())
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            if self.conf.Human_reward:
                episode_reward += (reward - human_inspire)
            else:
                episode_reward += reward
            if self.conf.epsilon_anneal_scale == 'step':
                for i in range(self.conf.n_agents):
                    epsilon[i] = epsilon[i] - self.anneal_epsilon[i] if epsilon[i] > self.end_epsilon[i] else epsilon[i]
        if any(information['crashed']):
            win_tag = False
        else:
            win_tag = True
        obs = self.obs_trans(next_obs)
        state = information["state"].flatten()
        o.append(obs)
        s.append(state)
        o_ = o[1:]
        s_ = s[1:]
        o = o[:-1]
        s = s[:-1]
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.get_avail_agent_actions(next_obs, agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_ = avail_u[1:]
        avail_u = avail_u[:-1]

        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_.append(np.zeros((self.n_agents, self.obs_shape)))
            s_.append(np.zeros(self.state_shape))
            z.append(np.zeros(self.conf.noise_d))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(
            o=o.copy(),
            s=s.copy(),
            u=u.copy(),
            r=r.copy(),
            o_=o_.copy(),
            s_=s_.copy(),
            z=z.copy(),
            avail_u=avail_u.copy(),
            avail_u_=avail_u_.copy(),
            u_onehot=u_onehot.copy(),
            padded=padded.copy(),
            terminated=terminate.copy()
        )

        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.start_epsilon = epsilon
        if evaluate:
            return episode, episode_reward / (step + 1e-9), win_tag, np.mean(np.array(speeds))
        else:
            return episode, episode_reward / (step + 1e-9), win_tag

    def obs_trans(self, next_obs):
        result = next_obs[0].reshape(1, -1)
        for i in range(1, len(next_obs)):
            obs_data = next_obs[i]
            result = np.concatenate((result, obs_data.reshape(1, -1)), axis=0)
        return copy.deepcopy(result)

    def get_avail_agent_actions(self, obs, id):
        return np.ones(self.n_actions)

    def speed_reward(self, obs, info, mul_weight, reward_speed_range=np.array([0, 20]), speed_weight=0.4):
        car_num = len(mul_weight)
        scaled_speeds = np.zeros(car_num)
        for i in range(car_num):
            forward_speed = info['speed'][i]  # * np.cos(obs_data1[0, 5])
            scaled_speed = np.clip(self.lmap(forward_speed, reward_speed_range, [0, 1]), 0, 1)
            crasheds = info['crashed']
            scaled_speeds[i] = speed_weight * scaled_speed + (-1) * float(crasheds[i])
        reward = np.dot(scaled_speeds, mul_weight)
        normalization_reward = self.lmap(reward, [-1, speed_weight], [0, 1])
        normalization_reward *= float(info['on_road_reward'])

        return normalization_reward

    def lmap(self, v: float, x, y) -> float:
        """Linear map of value v with range x to desired range y."""
        return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])

    def to_close(self, ttc):
        if sum(ttc) == 0:
            return False
        pri_agent = np.where(ttc == ttc[ttc > 0].min())[0]
        for i in range(len(ttc)):
            if i != pri_agent:
                if 0 < ttc[i] < 2 and (ttc[i] - ttc[pri_agent]) < 0.5:
                    return True
        return False


class ReplayBuffer:
    def __init__(self, conf):
        self.conf = conf
        self.episode_limit = conf.episode_limit
        self.n_actions = conf.n_actions
        self.n_agents = conf.n_agents
        self.state_shape = conf.state_shape
        self.obs_shape = conf.obs_shape
        self.size = conf.buffer_size

        self.current_idx = 0
        self.current_size = 0

        self.buffers = {'o': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape], dtype=np.float32),
                        'u': np.empty([self.size, self.episode_limit, self.n_agents, 1]),
                        's': np.empty([self.size, self.episode_limit, self.state_shape], dtype=np.float32),
                        'r': np.empty([self.size, self.episode_limit, 1]),
                        'o_': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape],
                                       dtype=np.float32),
                        's_': np.empty([self.size, self.episode_limit, self.state_shape], dtype=np.float32),
                        'z': np.empty([self.size, self.episode_limit, self.conf.noise_d], dtype=np.float32),  # 和s_绑定
                        'avail_u': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'avail_u_': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'u_onehot': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'padded': np.empty([self.size, self.episode_limit, 1]),
                        'terminated': np.empty([self.size, self.episode_limit, 1]),
                        }
        self.lock = threading.Lock()
        self.serial = np.zeros(self.size)
        print("Replay Buffer inited!")

    def store_episode(self, episode_batch):
        batch_size = episode_batch['o'].shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            self.buffers['o'][idxs] = episode_batch['o']
            self.buffers['u'][idxs] = episode_batch['u']
            self.buffers['s'][idxs] = episode_batch['s']
            self.buffers['r'][idxs] = episode_batch['r']
            self.buffers['o_'][idxs] = episode_batch['o_']
            self.buffers['s_'][idxs] = episode_batch['s_']
            self.buffers['z'][idxs] = episode_batch['z']
            self.buffers['avail_u'][idxs] = episode_batch['avail_u']
            self.buffers['avail_u_'][idxs] = episode_batch['avail_u_']
            self.buffers['u_onehot'][idxs] = episode_batch['u_onehot']
            self.buffers['padded'][idxs] = episode_batch['padded']
            self.buffers['terminated'][idxs] = episode_batch['terminated']
            self.serial[idxs] = np.random.rand(batch_size)

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            if self.conf.random_buffer:
                overflow = inc - (self.size - self.current_idx)
                idx_a = np.arange(self.current_idx, self.size)
                sorted_indices = np.argsort(self.serial)
                idx_b = np.sort(sorted_indices[:overflow])
                idx = np.concatenate([idx_a, idx_b])
            else:
                overflow = inc - (self.size - self.current_idx)
                idx_a = np.arange(self.current_idx, self.size)
                idx_b = np.arange(0, overflow)
                idx = np.concatenate([idx_a, idx_b])
                self.current_idx = overflow
        else:
            if self.conf.random_buffer:
                sorted_indices = np.argsort(self.serial)
                idx = np.sort(sorted_indices[:inc])
            else:
                idx = np.arange(0, inc)
                self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
