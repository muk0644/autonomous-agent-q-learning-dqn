import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

class ShariqQuestEnv(gym.Env):
    """
    Custom environment where Shariq tries to retrieve the Magic Stone while avoiding traps in a maze with time constraints.
    
    Attributes:
        grid_size (int): The size of the grid (grid_size x grid_size).
        max_steps (int): The maximum number of steps allowed per episode.
        current_step (int): The current step count in the episode.
        agent_state (np.ndarray): The current position of Shariq (the agent).
        goal_state (np.ndarray): The position of the Magic Stone (goal).
        danger_zones (list): List of coordinates representing the positions of traps.
        obstacles (list): List of coordinates representing the positions of barriers.
        action_space (gym.spaces.Discrete): The action space (up, down, left, right).
        observation_space (gym.spaces.Box): The observation space (position of the agent).
        fig (plt.Figure): Matplotlib figure for rendering.
        ax (plt.Axes): Matplotlib axes for rendering.
    """
    def __init__(self, grid_size=7, max_steps=50):
        super(ShariqQuestEnv, self).__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0
        self.agent_state = np.array([0, 0])  # Initial position of Shariq
        self.goal_state = np.array([grid_size-1, grid_size-1])  # Position of the Magic Stone
        self.danger_zones = [
            np.array([1, 1]), np.array([2, 2]), np.array([4, 4]), np.array([5, 5]), 
            np.array([1, 3]), np.array([2, 5]), np.array([3, 6]), np.array([5, 2])
        ]
        self.obstacles = [
            np.array([3, 1]), np.array([3, 2]), np.array([3, 3]), np.array([3, 4]), np.array([3, 5])
        ]
        
        # Action space: up, down, left, right
        self.action_space = gym.spaces.Discrete(4)
        
        # Observation space: agent's position
        self.observation_space = gym.spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32)
        
        self.fig, self.ax = plt.subplots()
        plt.show(block=False)
    
    def reset(self):
        """
        Reset the environment to the initial state.
        
        Returns:
            np.ndarray: The initial state of the agent.
        """
        self.agent_state = np.array([0, 0])  # Reset Shariq's position
        self.current_step = 0  # Reset step counter
        return self.agent_state
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action (int): The action taken by the agent.
        
        Returns:
            tuple: The new state, reward, done flag, and info dictionary.
        """
        self.current_step += 1
        new_state = self.agent_state.copy()
        
        # Update state based on action
        if action == 0 and self.agent_state[1] < self.grid_size - 1:  # up
            new_state[1] += 1
        elif action == 1 and self.agent_state[1] > 0:  # down
            new_state[1] -= 1
        elif action == 2 and self.agent_state[0] > 0:  # left
            new_state[0] -= 1
        elif action == 3 and self.agent_state[0] < self.grid_size - 1:  # right
            new_state[0] += 1

        # Update agent state only if the new state is not an obstacle
        if not any(np.array_equal(new_state, ob) for ob in self.obstacles):
            self.agent_state = new_state

        reward = -0.5  # Default reward for each step
        done = False
        
        # Check if Shariq has reached the goal (Magic Stone)
        if np.array_equal(self.agent_state, self.goal_state):
            reward = 10 + (self.max_steps - self.current_step)  # Bonus for reaching the goal faster
            done = True
        
        # Check if Shariq has encountered a trap
        if any(np.array_equal(self.agent_state, dz) for dz in self.danger_zones):
            reward = -1
            done = True

        # Check if max steps have been reached
        if self.current_step >= self.max_steps:
            done = True

        # Calculate distance to the goal
        distance_to_goal = np.linalg.norm(self.goal_state - self.agent_state)
        info = {"Distance to Goal": distance_to_goal}

        return self.agent_state, reward, done, info
    
    def render(self):
        """
        Render the environment.
        """
        self.ax.clear()
        self.ax.plot(self.agent_state[0], self.agent_state[1], "ro", label='Shariq')  # Shariq
        self.ax.plot(self.goal_state[0], self.goal_state[1], "g*", label='Magic Stone')  # Magic Stone
        for dz in self.danger_zones:
            self.ax.plot(dz[0], dz[1], "bs", label='Trap')  # Traps
        for ob in self.obstacles:
            self.ax.plot(ob[0], ob[1], "ks", label='Barrier')  # Barriers

        self.ax.set_xlim(-1, self.grid_size)
        self.ax.set_ylim(-1, self.grid_size)
        self.ax.set_aspect("equal")
        self.ax.legend()
        plt.pause(0.05)
    
    def close(self):
        """
        Close the rendering window.
        """
        plt.close()

# Main block to run the environment for testing purposes
if __name__ == "__main__":
    env = ShariqQuestEnv()
    state = env.reset()
    print("Initial state:", state)
    
    # Run with Random Agent for testing
    for _ in range(100):
        action = env.action_space.sample()  # Take a random action
        state, reward, done, info = env.step(action)
        env.render()
        print(f"State: {state}, Reward: {reward}, Done: {done}, Info: {info}")
        time.sleep(0.25)  # Add a delay to slow down the rendering
        if done:
            if reward > 0:
                print("Shariq retrieved the Magic Stone!")
            else:
                print("Shariq was caught by a Trap!")
            break
    
    env.close()
