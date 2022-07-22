import gym
import numpy as np
import torch
import torch.nn as nn
from tianshou.utils.net.discrete import NoisyLinear
from torch.distributions import Categorical

import tianshou as ts
from tianshou.utils.net.common import Net
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.policy import RainbowPolicy
from tianshou.trainer import offpolicy_trainer

task = 'CartPole-v1'
lr, epoch, batch_size = 1e-3, 10, 128
train_num, test_num = 8, 8
gamma, n_step, target_freq = 0.99, 4, 500
buffer_size = 20000
eps_train, eps_test = 1, 0.005
eps_train_final = 0.05
step_per_collect = 16
step_per_epoch = step_per_collect * 2000
hidden_sizes = (128, 128)
num_atoms = 51
alpha = 0.5
beta = 0.4
beta_final = 1
beta_anneal_step = 5000000

def make_env(render=False):
    env = gym.make(task, render_mode='human' if render else None)
    env._max_episode_steps = 10000
    return env

train_envs = ts.env.DummyVectorEnv([make_env for _ in range(train_num)])
test_envs = ts.env.DummyVectorEnv([make_env for _ in range(test_num)])

env = make_env(render=True)
state_shape = env.observation_space.shape
action_shape = env.action_space.n
Q_param = V_param = {"hidden_sizes": hidden_sizes, "linear_layer": NoisyLinear}
net = Net(state_shape=state_shape, action_shape=action_shape, num_atoms=num_atoms,
          hidden_sizes=hidden_sizes, dueling_param=(Q_param, V_param), softmax=True)
optim = torch.optim.Adam(net.parameters(), lr=lr)

policy = RainbowPolicy(net, optim, gamma, estimation_step=n_step, num_atoms=num_atoms, target_update_freq=target_freq)
# buffer = VectorReplayBuffer(buffer_size, train_num, ignore_obs_next=True, save_only_last_obs=False)
buffer = PrioritizedVectorReplayBuffer(
    buffer_size, buffer_num=train_num, ignore_obs_next=True, save_only_last_obs=False, alpha=alpha, beta=beta)
train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
test_collector = Collector(policy, test_envs, exploration_noise=True)  # because DQN uses epsilon-greedy method

train_collector.collect(n_step=batch_size * train_num)


def train_fn(epoch, env_step):
    # nature DQN setting, linear decay in the first 1M steps
    if env_step <= 1e6:
        eps = eps_train - env_step / 1e6 * \
              (eps_train - eps_train_final)
    else:
        eps = eps_train_final
    policy.set_eps(eps)
    if env_step <= beta_anneal_step:
        beta_ = beta - env_step / beta_anneal_step * (beta - beta_final)
    else:
        beta_ = beta_final
    buffer.set_beta(beta_)


result = offpolicy_trainer(
    policy, train_collector, test_collector, epoch, step_per_epoch, step_per_collect,
    test_num, batch_size, update_per_step=1 / step_per_collect,
    train_fn=train_fn, test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
    stop_fn=lambda mean_rewards: mean_rewards >= 100000, test_in_train=False)

print(f'Finished training! Use {result["duration"]}')

# torch.save(policy.state_dict(), 'pg.pth')
# policy.load_state_dict(torch.load('pg.pth'))

policy.eval()
policy.set_eps(eps_test)
collector = Collector(policy, env, exploration_noise=True)
collector.collect(n_episode=1, render=1 / 35)
