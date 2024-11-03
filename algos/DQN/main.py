import torch
import gymnasium as gym
import numpy as np
import argparse

from DQN import QLearningAgent, ReplayBuffer

seed = 30

def train():
    # Environment
    env = gym.make("LunarLander-v2")
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)

    # Hyperparameters
    obs_space_dims = wrapped_env.observation_space.shape[0]
    action_space_dims = wrapped_env.action_space.n
    hidden_size = 32
    learning_rate = 1e-4
    gamma = 0.99
    epsilon = 0.1

    # Agent
    agent = QLearningAgent(obs_space_dims, hidden_size, action_space_dims, learning_rate, gamma, epsilon)

    # Training
    buffer_length = int(1e4)
    batch_size = 128
    total_num_episodes = int(1e4)

    replay_buffer = ReplayBuffer(buffer_length)

    for episode in range(total_num_episodes):
        state, info = wrapped_env.reset(seed=seed)
        done = False
        while not done:
            action = agent.get_action(state, training=True)
            next_state, reward, terminated, truncated, info = wrapped_env.step(action)
            done = terminated or truncated
            replay_buffer.append(state, action, reward, next_state, done)
            state = next_state

            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                agent.train(batch)
        
        if (episode + 1) % 100 == 0:
            agent.update_target_network()
            print(f"Episode: {episode + 1}, Avg reward: {np.mean(wrapped_env.return_queue)}")
        
        if (episode + 1) % 1000 == 0:
            torch.save(agent.q_net.state_dict(), f"DQN_lunarlander_{episode+1}_episode.pth")
    wrapped_env.close()
    


def test():
    env = gym.make("LunarLander-v2", render_mode="human")

    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.n
    hidden_size = 32

    agent = QLearningAgent(obs_space_dims, hidden_size, action_space_dims)
    agent.q_net.load_state_dict(torch.load("weights/DQN_lunarlander.pth", weights_only=True))

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