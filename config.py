import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.io import savemat


class Config:
    def __init__(self, seed, flag=False):
        self.return_m = None
        self.use_rule = False
        self.n_agents = None
        self.obs_shape = None
        self.episode_limit = None
        self.seed = None
        self.state_shape = None
        self.n_actions = None
        self.load_num = None
        self.mix_network = None
        self.train = True
        self.cuda = True
        self.seed = seed
        self.test_flag = flag
        # train setting
        self.last_action = False
        self.reuse_network = True
        self.use_id = False
        self.attention = True
        self.noise_d = 16
        self.n_epochs = 20000  # 20000
        self.evaluate_epoch = 100
        self.evaluate_per_epoch = 200
        self.batch_size = 1500  # 32
        self.buffer_size = 15000
        self.save_frequency = 500000  # 5000
        self.n_eposodes = 3
        self.train_steps = 10
        self.jt_w = 0.1
        self.noopt_w = 0.001
        self.gamma = 0.8
        self.grad_norm_clip = 10  # prevent gradient explosion
        self.update_target_params = 50  # 200
        self.result_dir = './results/'
        self.load_model = False
        self.run_directory = "./record/"
        self.random_buffer = False
        self.ex_model_dir = "HI_874.pth"
        if self.cuda:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        # model structure
        # drqn net
        self.use_rnn = False
        self.drqn_hidden_dim = 512
        self.drqn_hidden_dim_2 = 128
        ###########################
        # qmix net
        # input: (batch_size, n_agents, qmix_hidden_dim)
        self.qmix_hidden_dim = 64
        self.two_hyper_layers = False
        self.hyper_hidden_dim = 64
        self.discrim_size = 128
        self.discrim_layers = 2
        self.entropy_scaling = 0.5
        self.model_dir = './vdn/'
        self.optimizer = "ADAM"  # "RMS"
        self.learning_rate = 5e-4
        self.lr = 1e-3
        # epsilon greedy
        self.start_epsilon = 1
        self.end_epsilon = 0.05
        self.anneal_steps = 50000  # 50000
        # self.anneal_epsilon = np.zeros(2)
        # self.anneal_epsilon[0] = (self.start_epsilon - self.end_epsilon) / self.anneal_steps
        # self.anneal_epsilon[1] = (self.start_epsilon - self.end_epsilon) / self.anneal_steps
        self.anneal_epsilon = (self.start_epsilon - self.end_epsilon) / self.anneal_steps
        self.epsilon_anneal_scale = 'step'
        self.Human_reward = False
        self.perfect = False
        self.Monotonically_decreasing = False
        self.de_f = 0.9999
        self.w = 1
        self.pw = 0.4
        self.nw = 0.1
        self.wqmix_w = 0.5
        self.mi = 0.1
        self.bandit_buffer = 5000
        self.bandit_iters = 20
        self.bandit_batch = 32
        #
        self.human_takeover = True
        self.real_human_control = True

    def set_env_info(self, env_info):
        self.n_actions = env_info.get("n_actions", 0)
        self.state_shape = env_info.get("state_shape", 0)
        self.obs_shape = env_info.get("obs_shape", 0)
        self.episode_limit = env_info.get("episode_limit", 0)
        self.load_num = env_info.get("num", 0)
        self.load_model = env_info.get("load_model", False)
        self.mix_network = env_info.get("mix_network", "Qmix")
        indices = torch.arange(self.episode_limit, dtype=torch.float32)
        self.return_m = self.gamma ** indices


def evaluate(conf, rollout_worker):
    # print("="*15, " evaluating ", "="*15)
    win_num = 0
    episode_rewards = []
    speeds = []
    for epoch in range(conf.evaluate_epoch):
        _, episode_reward, win_tag, speed = rollout_worker.generate_episode(epoch, evaluate=True)
        episode_rewards.append(episode_reward)
        speeds.append(speed)
        if win_tag:
            win_num += 1
    episode_rewards = np.array(episode_rewards)
    speeds = np.array(speeds)
    mean_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    mean_episode_reward = np.mean(episode_rewards)
    std_episode_reward = np.std(episode_rewards)
    return win_num / conf.evaluate_epoch, mean_speed, std_speed, mean_episode_reward, std_episode_reward

def show_curves(conf, win_rates, episode_rewards):
    np.save(conf.result_dir + '/win_rates', win_rates)
    np.save(conf.result_dir + '/episode_rewards', episode_rewards)
    savemat(conf.result_dir + '/result.mat', {"win_rates": win_rates, 'episode_rewards': episode_rewards})
