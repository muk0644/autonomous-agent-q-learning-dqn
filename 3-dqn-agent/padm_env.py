import gymnasium as gym
import numpy as np
import pygame
import sys

class ShariqQuestEnv(gym.Env):
    """
    Custom environment where Shariq tries to retrieve the Magic Stone while avoiding traps in a maze with time constraints.
    
    Attributes:
        grid_size (int): The size of the grid (grid_size x grid_size).
        cell_size (int): The size of each cell in the grid (for rendering).
        state (np.ndarray): The current position of Shariq (the agent).
        reward (int): The accumulated reward.
        info (dict): Additional information about the environment.
        goal (np.ndarray): The position of the Magic Stone (goal).
        done (bool): Whether the episode is done.
        hell_states (list): List of coordinates representing the positions of traps.
        obstacles (list): List of coordinates representing the positions of barriers.
        action_space (gym.spaces.Discrete): The action space (up, down, left, right).
        observation_space (gym.spaces.Box): The observation space (position of the agent).
    """
    def __init__(self, grid_size=7, goal_coordinates=(0, 6), render_mode=False) -> None:
        super(ShariqQuestEnv, self).__init__()
        self.grid_size = grid_size
        self.cell_size = 100
        self.state = np.array([6, 0])  # Agent starts at (6, 0)
        self.reward = 0
        self.info = {}
        self.goal = np.array(goal_coordinates)
        self.done = False
        self.hell_states = []
        self.obstacles = []
        self.render_mode = render_mode  # Track render mode

        # Define the action space: Up, Down, Left, Right
        self.action_space = gym.spaces.Discrete(4)

        # Define the observation space: agent's position
        self.observation_space = gym.spaces.Box(
            low=0, high=grid_size-1, shape=(2,), dtype=np.int32)

        if render_mode:
            # Initialize the window for rendering:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.cell_size * self.grid_size, self.cell_size * self.grid_size))

    def reset(self):
        """
        Reset the environment to the initial state.
        
        Returns:
            np.ndarray: The initial state of the agent.
            dict: Additional information about the environment.
        """
        self.state = np.array([6, 0])  # Agent starts at (6, 0)
        self.done = False
        self.reward = 0

        # Calculate and store the distance to the goal
        self.info["Distance to goal"] = np.sqrt(
            (self.state[0] - self.goal[0]) ** 2 +
            (self.state[1] - self.goal[1]) ** 2
        )

        return self.state, self.info

    def add_hell_states(self, hell_state_coordinates):
        """Add hell states (traps) to the environment."""
        self.hell_states.append(np.array(hell_state_coordinates))

    def add_obstacles(self, obstacle_coordinates):
        """Add obstacles (barriers) to the environment."""
        self.obstacles.append(np.array(obstacle_coordinates))

    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action (int): The action taken by the agent.
        
        Returns:
            tuple: The new state, reward, done flag, and info dictionary.
        """
        # Define possible actions:
        if action == 0 and self.state[0] > 0:  # Up
            new_state = self.state + [-1, 0]
        elif action == 1 and self.state[0] < self.grid_size - 1:  # Down
            new_state = self.state + [1, 0]
        elif action == 2 and self.state[1] < self.grid_size - 1:  # Right
            new_state = self.state + [0, 1]
        elif action == 3 and self.state[1] > 0:  # Left
            new_state = self.state + [0, -1]
        else:
            new_state = self.state

        # Update agent state only if the new state is not an obstacle
        if not any(np.array_equal(new_state, ob) for ob in self.obstacles):
            self.state = new_state

        # Define the reward structure:
        if np.array_equal(self.state, self.goal):  # Check goal condition
            self.reward += 10
            self.done = True
        elif any(np.array_equal(self.state, each_hell) for each_hell in self.hell_states):  # Check hell-states
            self.reward += -1
            self.done = True
        else:  # Every other state
            self.reward += 0
            self.done = False

        # Update distance to goal in info dictionary
        self.info["Distance to goal"] = np.sqrt(
            (self.state[0] - self.goal[0]) ** 2 +
            (self.state[1] - self.goal[1]) ** 2
        )

        if self.render_mode:
            self.render()

        return self.state, self.reward, self.done, self.info

    def render(self, mode='human'):
        """
        Render the environment.
        """
        if not hasattr(self, 'screen'):
            return

        # Code for closing the window:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Make the background white
        self.screen.fill((255, 255, 255))

        # Draw grid lines:
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                grid = pygame.Rect(
                    y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), grid, 1)

        # Draw the goal state:
        goal = pygame.Rect(self.goal[1] * self.cell_size, self.goal[0]
                           * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 255, 0), goal)

        # Draw the hell states:
        for each_hell in self.hell_states:
            hell = pygame.Rect(
                each_hell[1] * self.cell_size, each_hell[0] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (255, 0, 0), hell)

        # Draw the obstacles:
        for each_obstacle in self.obstacles:
            obstacle = pygame.Rect(
                each_obstacle[1] * self.cell_size, each_obstacle[0] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (0, 0, 255), obstacle)

        # Draw the agent:
        agent = pygame.Rect(self.state[1] * self.cell_size, self.state[0]
                            * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 0, 0), agent)

        # Update contents on the window:
        pygame.display.flip()

    def close(self):
        if hasattr(self, 'screen'):
            pygame.quit()

def create_env(goal_coordinates, hell_state_coordinates, obstacle_coordinates, render_mode=False):
    """
    Create an instance of the ShariqQuest environment with specified goal, hell states, and obstacles.
    
    Args:
        goal_coordinates (tuple): The coordinates of the goal.
        hell_state_coordinates (list): A list of coordinates for the hell states.
        obstacle_coordinates (list): A list of coordinates for the obstacles.
        render_mode (bool): Whether to enable rendering.
    
    Returns:
        ShariqQuestEnv: The created environment instance.
    """
    env = ShariqQuestEnv(goal_coordinates=goal_coordinates, render_mode=render_mode)

    for hell_state in hell_state_coordinates:
        env.add_hell_states(hell_state_coordinates=hell_state)

    for obstacle in obstacle_coordinates:
        env.add_obstacles(obstacle_coordinates=obstacle)

    return env
