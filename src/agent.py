# Reinforcement Learning Agent Implementation (Pong Environment)

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma

    def choose_action(self, state):
        with torch.no_grad():
            values = self.model(torch.FloatTensor(state).unsqueeze(0))
            return torch.argmax(values).item()


if __name__ == '__main__':
    env = gym.make('ALE/PongDeterministic-v5', render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = Agent(state_size, action_size)

    state = env.reset()[0]  # Gymnasium API
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action) # Gymnasim API
        done = terminated or truncated
        env.render()
        state = next_state

    env.close()