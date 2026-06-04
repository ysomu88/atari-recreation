import torch
import numpy as np
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from environment.environment import make_atari_env
from agent.agent import DQNAgent

def train(config):
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = make_atari_env(env_id=config['env_id'], seed=config['seed'])
    agent = DQNAgent(n_actions=env.n_actions, device=device, lr=config['lr'], gamma=config['gamma'], batch_size=config['batch_size'], target_update_freq=config['target_update_freq'])
    
    writer = SummaryWriter(log_dir=config['log_dir'])
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    global_step = 0
    episode = 0
    episode_reward = 0.0
    episode_losses = []
    
    obs, info = env.reset()
    
    while global_step < config['total_steps']:
        action = agent.select_action(obs, global_step)
        obs_next, reward, terminated, truncated, info = env.step(action)
        
        agent.store(obs, action, reward, obs_next, terminated or truncated)
        
        loss = agent.optimise()
        if loss > 0:
            episode_losses.append(loss)
            writer.add_scalar('Loss/step', loss, global_step)

        agent.maybe_update_target(global_step)
        
        obs = obs_next
        global_step += 1
        episode_reward += reward
        
        if terminated or truncated:
            episode += 1
            avg_loss = np.mean(episode_losses) if episode_losses else 0.0
            
            writer.add_scalar('Reward/episode', episode_reward, episode)
            writer.add_scalar('Epsilon/episode', agent.epsilon_schedule.value(global_step), episode)
            writer.add_scalar('Loss/episode', avg_loss, episode)
            
            if episode % config['print_every'] == 0:
                print(f"Step: {global_step}, Episode: {episode}, Reward: {episode_reward}, Epsilon: {agent.epsilon_schedule.value(global_step)}, Avg Loss: {avg_loss}")
            
            if episode % 100 == 0:
                torch.save(agent.online_net.state_dict(), os.path.join(config['checkpoint_dir'], f'dqn_episode_{episode}.pth'))
            
            episode_reward = 0.0
            episode_losses = []
            obs, info = env.reset()
    
    writer.close()
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent on Atari environment.")
    parser.add_argument("--total_steps", type=int, default=500000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default='runs/pong_dqn')
    parser.add_argument("--checkpoint_dir", type=str, default='checkpoints')
    
    args = parser.parse_args()
    config = {
        "env_id": 'ALE/Pong-v5',
        "seed": args.seed,
        "total_steps": args.total_steps,
        "batch_size": 32,
        "lr": args.lr,
        "gamma": 0.99,
        "target_update_freq": 1000,
        "log_dir": args.log_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "print_every": 10
    }
    
    train(config)