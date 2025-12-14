import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def train_q_learning(env, no_episodes, epsilon, epsilon_min, epsilon_decay, alpha, gamma, q_table_save_path="q_table.npy"):
    """
    Train a Q-learning agent.

    Args:
        env (gym.Env): The environment to train in.
        no_episodes (int): Number of training episodes.
        epsilon (float): Initial exploration rate.
        epsilon_min (float): Minimum exploration rate.
        epsilon_decay (float): Decay rate for exploration.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        q_table_save_path (str): Path to save the Q-table.
    """
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    for episode in range(no_episodes):
        state, _ = env.reset()
        state = tuple(state)
        total_reward = 0

        while True:
            # Exploration vs. Exploitation
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            next_state, reward, done, _ = env.step(action)
            env.render()

            next_state = tuple(next_state)
            total_reward += reward

            # Q-learning update rule
            q_table[state][action] = q_table[state][action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

            state = next_state

            if done:
                break

        # Epsilon decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    env.close()
    print("Training finished.\n")
    np.save(q_table_save_path, q_table)
    print("Saved the Q-table.")

def visualize_q_table(hell_state_coordinates=[(3, 2), (2, 3), (4, 4), (3, 5)], goal_coordinates=(0, 6), obstacle_coordinates=[(3, 1), (3, 3), (4, 3), (5, 3), (1, 5)], actions=["Up", "Down", "Right", "Left"], q_values_path="q_table.npy"):
    """
    Visualize the Q-table.

    Args:
        hell_state_coordinates (list): A list of coordinates for the hell states.
        goal_coordinates (tuple): The coordinates of the goal.
        obstacle_coordinates (list): A list of coordinates for the obstacles.
        actions (list): A list of action names.
        q_values_path (str): Path to the Q-table.
    """
    try:
        q_table = np.load(q_values_path)
        _, axes = plt.subplots(1, 4, figsize=(20, 5))

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()
            mask = np.zeros_like(heatmap_data, dtype=bool)
            mask[goal_coordinates] = True

            for hell in hell_state_coordinates:
                mask[hell] = True

            for obstacle in obstacle_coordinates:
                mask[obstacle] = True

            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar=False, mask=mask, annot_kws={"size": 9})

            ax.text(goal_coordinates[1] + 0.5, goal_coordinates[0] + 0.5, 'G', color='green', ha='center', va='center', weight='bold', fontsize=14)

            for hell in hell_state_coordinates:
                ax.text(hell[1] + 0.5, hell[0] + 0.5, 'H', color='red', ha='center', va='center', weight='bold', fontsize=14)

            for obstacle in obstacle_coordinates:
                ax.text(obstacle[1] + 0.5, obstacle[0] + 0.5, 'O', color='blue', ha='center', va='center', weight='bold', fontsize=14)

            ax.set_title(f'Action: {action}')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")
