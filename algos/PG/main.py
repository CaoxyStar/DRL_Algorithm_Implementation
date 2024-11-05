import torch
import gymnasium as gym
import numpy as np
import argparse

from PG import PG_Agent

seed = 30

def train():
    # Environment
    env = gym.make("InvertedPendulum-v4")
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)

    # Hyperparameters
    obs_space_dims = wrapped_env.observation_space.shape[0]
    action_space_dims = wrapped_env.action_space.shape[0]
    hidden_size = 32
    learning_rate = 1e-4
    gamma = 0.99

    # Agent
    agent = PG_Agent(obs_space_dims, hidden_size, action_space_dims, learning_rate, gamma)

    # Training
    total_num_episodes = int(1e4)

    for episode in range(total_num_episodes):
        state, info = wrapped_env.reset(seed=seed)
        done = False
        while not done:
            action = agent.get_action(state, training=True)
            next_state, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)
            done = terminated or truncated
            state = next_state
        
        # Update
        agent.train()
        
        # Logging and saving
        if (episode + 1) % 100 == 0:
            print(f"Episode: {episode + 1}, Avg reward: {np.mean(wrapped_env.return_queue)}")
        
        if (episode + 1) % 1000 == 0:
            torch.save(agent.net.state_dict(), f"PolicyGradient_Inverted_Pendulum_{episode+1}_episode.pth")
    
    wrapped_env.close()
    

def test():
    # Environment
    env = gym.make("InvertedPendulum-v4", render_mode="human")

    # Agent
    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.shape[0]
    hidden_size = 32

    agent = PG_Agent(obs_space_dims, hidden_size, action_space_dims)
    agent.net.load_state_dict(torch.load("weights/PolicyGradient_Inverted_Pendulum.pth", weights_only=True))

    # Testing
    state, info = env.reset(seed=seed)
    for _ in range(1000):
        action = agent.get_action(state, training=False)
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            state, info = env.reset(seed=seed)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.test:
        test()
    else:
        print("Please specify --train or --test")