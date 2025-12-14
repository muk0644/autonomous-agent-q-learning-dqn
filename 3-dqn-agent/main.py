# main.py

# Imports:
import torch
import gymnasium as gym
from DQN_model import Qnet
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import ReplayBuffer, train
from padm_env import create_env

# User definitions:
train_dqn = True  # Enable training
test_dqn = True   # Enable testing
render = True     # Enable rendering

# Hyperparameters:
learning_rate = 0.005
gamma = 0.98
buffer_limit = 50_000
batch_size = 32
num_episodes = 50_000  # Number of episodes for training
max_steps = 10_000

# Main:
if train_dqn:
    env = create_env(goal_coordinates=(0, 6),
                     hell_state_coordinates=[(3, 2), (2, 3), (4, 4), (3, 5)],
                     obstacle_coordinates=[(3, 1), (3, 3), (4, 3), (5, 3), (1, 5)],
                     render_mode=render)

    # Initialize the Q Net and the Q Target Net
    q_net = Qnet(no_actions=env.action_space.n, no_states=env.observation_space.shape[0])
    q_target = Qnet(no_actions=env.action_space.n, no_states=env.observation_space.shape[0])

    # Load existing model weights if available
    try:
        q_net.load_state_dict(torch.load("dqn.pth"))
        q_target.load_state_dict(q_net.state_dict())
        print("Loaded existing model weights.")
    except FileNotFoundError:
        print("No existing model weights found. Starting from scratch.")

    # Initialize the Replay Buffer
    memory = ReplayBuffer(buffer_limit=buffer_limit)

    print_interval = 20
    episode_reward = 0.0
    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)

    rewards = []

    for n_epi in range(num_episodes):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%

        s, _ = env.reset()
        done = False

        for _ in range(max_steps):
            a = q_net.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, reward, done, _ = env.step(a)

            done_mask = 0.0 if done else 1.0

            memory.put((s, a, reward, s_prime, done_mask))
            s = s_prime

            episode_reward += reward

            if done:
                break

        if memory.size() > 2000:
            train(q_net, q_target, memory, optimizer, batch_size, gamma)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q_net.state_dict())
            print(f"n_episode: {n_epi}, Episode reward: {episode_reward}, n_buffer: {memory.size()}, eps: {epsilon}")

        rewards.append(episode_reward)
        episode_reward = 0.0

        if rewards[-10:] == [max_steps] * 10:
            break

    env.close()
    torch.save(q_net.state_dict(), "dqn.pth")

    plt.plot(rewards, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.legend()
    plt.savefig("training_curve.png")
    plt.show()

if test_dqn:
    print("Testing the trained DQN:")
    env = create_env(goal_coordinates=(0, 6),
                     hell_state_coordinates=[(3, 2), (2, 3), (4, 4), (3, 5)],
                     obstacle_coordinates=[(3, 1), (3, 3), (4, 3), (5, 3), (1, 5)],
                     render_mode=True)  # Enable rendering during testing

    dqn = Qnet(no_actions=env.action_space.n, no_states=env.observation_space.shape[0])
    dqn.load_state_dict(torch.load("dqn.pth"))

    for _ in range(10):
        s, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            action = dqn(torch.from_numpy(s).float())
            s_prime, reward, done, _ = env.step(action.argmax().item())
            s = s_prime

            episode_reward += reward

            if done:
                break
        print(f"Episode reward: {episode_reward}")

    env.close()
