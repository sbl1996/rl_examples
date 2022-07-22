import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

import tianshou as ts
from tianshou.utils.net.common import Net
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import PGPolicy
from tianshou.trainer import onpolicy_trainer

task = 'CartPole-v1'
lr, epoch, batch_size = 1e-3, 10, 256
train_num, test_num = 10, 10
gamma = 0.99
buffer_size = 20000
step_per_epoch, step_per_collect = 100000, 2048
repeat_per_collect = 1

def make_env(render=False):
    return gym.make(task, render_mode='human' if render else None).env

train_envs = ts.env.DummyVectorEnv([make_env for _ in range(train_num)])
test_envs = ts.env.DummyVectorEnv([make_env for _ in range(test_num)])

env = make_env(render=True)
state_shape = env.observation_space.shape
action_shape = env.action_space.n
net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[32, 32])
optim = torch.optim.Adam(net.parameters(), lr=lr)

dist = lambda x: Categorical(logits=x)

policy = PGPolicy(
    net, optim, dist, gamma, action_space=env.action_space)
train_collector = Collector(policy, train_envs, VectorReplayBuffer(buffer_size, train_num))
test_collector = Collector(policy, test_envs)

result = onpolicy_trainer(
    policy, train_collector, test_collector, epoch, step_per_epoch,
    repeat_per_collect, test_num, batch_size, step_per_collect=step_per_collect,
    stop_fn=lambda mean_rewards: mean_rewards >= 10000)

print(f'Finished training! Use {result["duration"]}')

# torch.save(policy.state_dict(), 'pg.pth')
# policy.load_state_dict(torch.load('pg.pth'))

policy.eval()
collector = ts.data.Collector(policy, env)
collector.collect(n_episode=1, render=1 / 35)
