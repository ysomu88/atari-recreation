import torch
import ale_py
import gymnasium as gym
import numpy as np
import argparse
from environment.wrappers import make_env
from agent.networks import NatureDQN


def evaluate(checkpoint_path, n_episodes=5, render=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make environment with rendering
    render_mode = "human" if render else None
    env = make_env(env_id="ALE/Pong-v5", render_mode=render_mode)

    # Load network
    n_actions = env.action_space.n
    net = NatureDQN(n_actions=n_actions).to(device)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Running {n_episodes} episodes...\n")

    rewards = []

    for episode in range(1, n_episodes + 1):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            obs_t = torch.tensor(
                np.array(obs), dtype=torch.float32
            ).unsqueeze(0).to(device) / 255.0

            with torch.no_grad():
                action = net(obs_t).argmax(dim=1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        rewards.append(episode_reward)
        print(f"Episode {episode}: Reward = {episode_reward}")

    env.close()
    print(f"\nAverage reward over {n_episodes} episodes: {np.mean(rewards):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    evaluate(args.checkpoint, n_episodes=args.episodes)