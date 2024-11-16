import torch
import gymnasium as gym
import numpy as np
import argparse

from PPO import PPO_Agent

seed = 30

def train():
    # Environment
    env = gym.make("LunarLander-v3")
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)

    # Hyperparameters
    obs_space_dims = wrapped_env.observation_space.shape[0]
    action_space_dims = wrapped_env.action_space.n
    hidden_size = 32
    actor_lr = 1e-4
    value_lr = 1e-3
    epsilon = 0.2

    # Agent
    agent = PPO_Agent(obs_space_dims, hidden_size, action_space_dims, actor_lr, value_lr, epsilon)

    # Training
    total_num_episodes = int(1e5)

    for episode in range(total_num_episodes):
        # Record List
        done = False
        states = []
        actions = []
        probs = []
        rewards = []

        state, info = wrapped_env.reset(seed=seed)
        while not done:
            action, action_prob = agent.get_action(state, training=True)
            next_state, reward, terminated, truncated, info = wrapped_env.step(action)
            
            states.append(state)
            actions.append(action)
            probs.append(action_prob)
            rewards.append(reward)

            done = terminated or truncated
            state = next_state
        
        # Calculate cumulative rewards
        g = 0
        gamma = 0.99
        returns = []

        for reward in reversed(rewards):
            g = reward + gamma * g
            returns.insert(0, g)

        # Updata
        agent.train(states, actions, probs, returns)
        
        # Logging and saving weights
        if (episode + 1) % 100 == 0:
            print(f"Episode: {episode + 1}, Avg reward: {np.mean(wrapped_env.return_queue)}")
        
        if (episode + 1) % 1000 == 0:
            torch.save(agent.actor_net.state_dict(), f"PPO_lunarlander_{episode+1}_episode.pth")

    wrapped_env.close()
    


def test():
    # Environment
    env = gym.make("LunarLander-v3", render_mode="human")

    # Agent
    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.n
    hidden_size = 32

    agent = PPO_Agent(obs_space_dims, hidden_size, action_space_dims)
    agent.actor_net.load_state_dict(torch.load("weights/PPO_lunarlander.pth", weights_only=True))

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