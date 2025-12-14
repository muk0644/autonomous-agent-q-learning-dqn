from padm_env import create_env
from q_learning import train_q_learning, visualize_q_table

# User definitions:
train = True
visualize_results = True

# Hyperparameters for Q-learning
learning_rate = 0.03
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.999
no_episodes = 50000

# Environment setup
goal_coordinates = (0, 6)
hell_state_coordinates = [(3, 2), (2, 3), (4, 4), (3, 5)]
obstacle_coordinates = [(3, 1), (3, 3), (4, 3), (5, 3), (1, 5)]

if train:
    # Create the environment
    env = create_env(goal_coordinates=goal_coordinates,
                     hell_state_coordinates=hell_state_coordinates,
                     obstacle_coordinates=obstacle_coordinates)

    # Train the Q-learning agent
    train_q_learning(env=env,
                     no_episodes=no_episodes,
                     epsilon=epsilon,
                     epsilon_min=epsilon_min,
                     epsilon_decay=epsilon_decay,
                     alpha=learning_rate,
                     gamma=gamma,
                     q_table_save_path="q_table.npy")

if visualize_results:
    # Visualize the Q-table
    visualize_q_table(hell_state_coordinates=hell_state_coordinates,
                      goal_coordinates=goal_coordinates,
                      obstacle_coordinates=obstacle_coordinates,
                      q_values_path="q_table.npy")
