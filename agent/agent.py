import torch
import torch.nn as nn
import numpy as np
import collections
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size, device):
        transitions = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*transitions)
        return {
            'obs': torch.tensor(np.array(obs), dtype=torch.uint8).float().div(255.0).to(device),
            'action': torch.tensor(actions, dtype=torch.long).to(device),
            'reward': torch.tensor(rewards, dtype=torch.float32).to(device),
            'next_obs': torch.tensor(np.array(next_obs), dtype=torch.uint8).float().div(255.0).to(device),
            'done': torch.tensor(dones, dtype=torch.float32).to(device),
        }

    def __len__(self):
        return len(self.buffer)


class EpsilonSchedule:
    def __init__(self, eps_start=1.0, eps_end=0.05, eps_decay_steps=100000):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

    def value(self, step):
        fraction = min(step / self.eps_decay_steps, 1.0)
        return self.eps_start + fraction * (self.eps_end - self.eps_start)


class DQNAgent:
    def __init__(self, n_actions, device, lr=1e-4, gamma=0.99, batch_size=32, target_update_freq=1000):
        from agent.networks import NatureDQN
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.online_net = NatureDQN(n_actions).to(device)
        self.target_net = NatureDQN(n_actions).to(device)
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.epsilon_schedule = EpsilonSchedule()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, obs, step):
        eps_threshold = self.epsilon_schedule.value(step)
        if random.random() < eps_threshold:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            obs_t = torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0).to(self.device) / 255.0
            return self.online_net(obs_t).argmax(dim=1).item()

    def store(self, obs, action, reward, next_obs, done):
        self.replay_buffer.push(obs, action, reward, next_obs, done)

    def optimise(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        batch = self.replay_buffer.sample(self.batch_size, self.device)
        obs = batch['obs']
        actions = batch['action']
        rewards = batch['reward']
        next_obs = batch['next_obs']
        dones = batch['done']

        q_values = self.online_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_obs).max(dim=1)[0]
        targets = rewards + self.gamma * (1.0 - dones) * next_q_values

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def maybe_update_target(self, step):
        if step > 0 and step % self.target_update_freq == 0:
            self.update_target()