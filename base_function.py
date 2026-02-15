import copy
import random
import numpy as np
import torch
import math
import gymnasium as gym
from torch.utils.data import DataLoader

from highway_env.envs import IntersectionEnv

negative_reward = -1


class my_env(gym.Env):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.env = make_env(self.conf)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space[0]
        grid_len = self.env.observation_space[0].shape[0] * len(self.env.observation_space)
        grid_seq = self.env.observation_space[0].shape[1]
        self.state_space = gym.spaces.Box(shape=[grid_len, grid_seq], low=-np.inf, high=np.inf, dtype=np.float32)
        self.set_mode("rgb_array")

    def step(self, actions):
        observation, reward, terminated, truncated, info = self.env.step(actions)
        info["state"] = np.vstack(observation)
        return tuple(observation), reward, terminated, truncated, info

    def reset(self, seed=None):
        if seed is None:
            seed = random.randint(1, 1000000000)
        observation, info = self.env.reset(seed=seed)
        info["state"] = np.vstack(observation)
        return tuple(observation), info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def set_mode(self, mode="human"):
        self.env.render_mode = mode

    def get_obs_shape(self):
        return np.prod(self.observation_space.shape)

    def get_state_shape(self):
        return np.prod(self.state_space.shape)

    def virtual_step(self, actions):
        reward, info = self.env.virtual_step(actions)
        info["ttc"] = info["ttc"] + 0 * np.random.normal(0, 0.0004, 4)
        info["agents_rewards"] = info["agents_rewards"] + 0 * np.random.normal(0, 0.0004, 4)
        return reward, info


def make_env(conf):
    env = IntersectionEnv()
    config = {
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "Kinematics",
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "absolute": True,
                "normalize": True,
                "order": "ok"
            }
        },
        "initial_vehicle_count": 0,
        "controlled_vehicles": conf.n_agents,
        "spawn_probability": 0.6,
        "action": {
            "type": "MultiAgentAction",
            "action_config": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": False,
                "target_speeds": np.linspace(0, 20, 3)
            },
        },
        "duration": 20,
        'screen_height': 800,
        'screen_width': 800,
        "scaling": 5.5 * 1.3,
        "centering_position": [0.5, 0.5],
        "reward_speed_range": [0, 20],
        "high_speed_reward": 1,
        "arrived_reward": 5,
        "collision_reward": -50,
        "test_time": conf.test_flag,
        "random": conf.random,
    }
    env.configure(config)
    env.reset(seed=conf.seed)
    return env


def lmap(v: float, x, y) -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


def set_seed(random_seed=11):
    np.set_printoptions(suppress=True)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(random_seed)
    else:
        torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_tensor(data):
    return torch.tensor(data, dtype=torch.float)


def to_array(data):
    return data.cpu().detach().numpy()


def IO_human(action, obs):
    # actions 是一个元组，用以表示限制
    done = True
    actions = (0, 1, 2)
    while done:
        try:
            print_direct(action, obs)
            action = input("请输入动作(0减速, 1保持, 2加速):")
            if action == 'q':
                action = None
                print("退出人类控制！")
                break
            elif action == 'o':
                action = 3
                print("接受智能体动作！")
                break
            else:
                action = int(action)
            if action in actions:  # 检查是否符合枚举范围
                done = False
            else:
                print('输入不合法，请重新输入！')
                done = True
        except ValueError:
            print('输入不合法，请重新输入！')
            done = True
    return action


def print_direct(action, obs):
    # 这里考虑输出信息，应该提供什么给专家考虑动作
    index = 0
    for i in range(1, obs.shape[0]):
        if obs[i, 0] == 1:
            index += 1
            print(
                f'观测到车辆{index}, 位置: x: {obs[i, 1] * 100}, y: {obs[i, 2] * 100}, 速度: vx: {obs[i, 3] * 20}, 速度: vy: {obs[i, 4] * 20}')
    print(
        f'当前智能体的位置为: x: {obs[0, 1] * 100}, y: {obs[0, 2] * 100}, vx: {obs[0, 3] * 20}, vy: {obs[0, 4] * 20}, 智能体的动作为:{action}, 等待专家决策...')


def action_dec(obs, info):
    action = np.ones(len(obs))
    ttc = np.array(info["ttc"])
    if sum(ttc) == 0:
        action = action * 2

    else:
        pri_agent = np.where(ttc == ttc[ttc > 0].min())[0]
        action[pri_agent] = 2
        for i in range(len(obs)):
            if i != pri_agent:
                if ttc[i] < 2 and (ttc[i] - ttc[pri_agent]) < 0.5:
                    action[i] = 0
            if info["out_flag"][i]:
                action[i] = 2
    return tuple(action)


def rule_dec(action, agent_id, obs, info):
    obs = obs.reshape(-1, 7)
    if info["out_flag"][agent_id]:
        action = 2
    ttc = np.array(info["ttc"])
    if sum(ttc) - ttc[agent_id] == 0:
        action = 2
    return action

