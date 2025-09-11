import os
import time
import numpy as np
import torch
from agent import Agents
from utils import RolloutWorker, ReplayBuffer
from base_function import set_seed, my_env
from config import Config, show_curves, evaluate

conf = Config(1902)  # seed


def train():
    ###
    conf.n_agents = 4
    conf.random = True
    conf.test_flag = False
    conf.w = 2
    conf.pw = 0.6
    conf.nw = 0.4
    conf.gamma = 0.8
    conf.n_eposodes = 1
    conf.train_steps = 3
    conf.drqn_hidden_dim = 512
    conf.drqn_hidden_dim_2 = 128
    conf.result_dir = "s0625/real_human"
    set_seed(random_seed=conf.seed)
    env = my_env(conf)
    obs_shape = env.get_obs_shape()
    state_shape = env.get_state_shape()
    env_info = {"n_actions": 3, "state_shape": state_shape, "obs_shape": obs_shape,
                "episode_limit": 20, "mix_network": "VDN"}
    conf.set_env_info(env_info)
    agents = Agents(conf)
    rollout_worker = RolloutWorker(env, agents, conf)
    buffer = ReplayBuffer(conf)

    # save plt and pkl
    save_path = conf.result_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    win_rates = []
    episode_rewards = []
    train_steps = 0
    train_reward_step = 0
    train_episode_rewards = 0
    train_win_num = 0
    start_time = time.time()
    max_speeds = -1
    max_rewards = -1
    std_speeds = 0
    std_rewards = 0
    max_rate = 0
    for epoch in range(conf.n_epochs):
        # print("train epoch: %d" % epoch)
        episodes = []
        for episode_idx in range(conf.n_eposodes):
            episode, train_episode_reward, win_tag = rollout_worker.generate_episode(episode_num=episode_idx,
                                                                                     render=False,
                                                                                     human_control=True,
                                                                                     human_id=0)
            if win_tag:
                train_win_num += 1
            train_reward_step += 1
            train_episode_rewards += train_episode_reward
            episodes.append(episode)
        episode_batch = episodes[0]
        episodes.pop(0)
        for episode in episodes:
            for key in episode_batch.keys():
                episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

        buffer.store_episode(episode_batch)
        for train_step in range(conf.train_steps):
            mini_batch = buffer.sample(
                min(buffer.current_size, conf.batch_size))
            agents.train(mini_batch, train_steps)
            train_steps += 1

        if (epoch > -1) and (epoch % conf.evaluate_per_epoch == 0):
            win_rate, mean_speed, std_speed, episode_reward, std_episode_reward = evaluate(conf, rollout_worker)
            win_rates.append(win_rate)
            episode_rewards.append(episode_reward)
            if win_rate > 0.96 and episode_reward > max_rewards:
                max_rate = win_rate
                max_rewards = episode_reward
                std_rewards = std_episode_reward
                max_speeds = mean_speed
                std_speeds = std_speed
                str_save = conf.result_dir + f"model_{int(win_rate * 1000)}.pth"
                torch.save(agents, str_save)
                print(str_save)
            print(f"train epoch: {epoch}, train rewards:{train_episode_rewards / (train_reward_step + 1e-9)},"
                  f"train_win:{train_win_num / (train_reward_step + 1e-9) * 100}%,\n"
                  f"evaluate win rate: {win_rate * 100}%, evaluate episode reward: {episode_reward}±{std_episode_reward}, "
                  f"evaluate speed: {mean_speed}±{std_speed}.\n"
                  f"The max rate is {max_rate}, max episode reward is {max_rewards}±{std_rewards}, max speed is {max_speeds}±{std_speeds}.")

            train_episode_rewards = 0
            train_reward_step = 0
            train_win_num = 0
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"The code ran for {elapsed_time} seconds\n")
            start_time = end_time

    show_curves(conf, win_rates, episode_rewards)


if __name__ == "__main__":
    if conf.train:
        train()
