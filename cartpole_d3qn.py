import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

import tianshou as ts
from tianshou.utils.net.common import Net
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer

task = 'CartPole-v1'
lr, epoch, batch_size = 1e-3, 10, 128
train_num, test_num = 8, 8
gamma, n_step, target_freq = 0.99, 4, 500
buffer_size = 20000
eps_train, eps_test = 0.1, 0.05
step_per_collect = 16
step_per_epoch = step_per_collect * 2000
hidden_sizes = (128, 128)

def make_env(render=False):
    env = gym.make(task, render_mode='human' if render else None)
    env._max_episode_steps = 10000
    return env

train_envs = ts.env.DummyVectorEnv([make_env for _ in range(train_num)])
test_envs = ts.env.DummyVectorEnv([make_env for _ in range(test_num)])

env = make_env(render=True)
state_shape = env.observation_space.shape
action_shape = env.action_space.n
Q_param = V_param = {"hidden_sizes": hidden_sizes}
net = Net(state_shape=state_shape, action_shape=action_shape,
          hidden_sizes=hidden_sizes, dueling_param=(Q_param, V_param))
optim = torch.optim.Adam(net.parameters(), lr=lr)

policy = DQNPolicy(net, optim, gamma, n_step, target_update_freq=target_freq)
train_collector = Collector(policy, train_envs, VectorReplayBuffer(buffer_size, train_num), exploration_noise=True)
test_collector = Collector(policy, test_envs, exploration_noise=True)  # because DQN uses epsilon-greedy method

train_collector.collect(n_step=batch_size * train_num)

result = offpolicy_trainer(
    policy, train_collector, test_collector, epoch, step_per_epoch, step_per_collect,
    test_num, batch_size, update_per_step=1 / step_per_collect,
    train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
    test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
    stop_fn=lambda mean_rewards: mean_rewards >= 100000)

print(f'Finished training! Use {result["duration"]}')

# torch.save(policy.state_dict(), 'pg.pth')
# policy.load_state_dict(torch.load('pg.pth'))

policy.eval()
policy.set_eps(eps_test)
collector = Collector(policy, env, exploration_noise=True)
collector.collect(n_episode=1, render=1 / 35)
