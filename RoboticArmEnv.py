import pygame
import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

class RoboticArmEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(RoboticArmEnv, self).__init__()

        self.steps_done = 0
        self.max_steps = 200
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(4,), dtype=np.float32)

        self.angle1 = 0.0
        self.angle2 = 0.0
        self.length1 = 1.0  # main arm
        self.length2 = 0.5  # upper arm
        self.prev_angle1 = self.angle1
        self.prev_angle2 = self.angle2
        self.end_effector_x = self.length1 * np.cos(self.angle1) + self.length2 * np.cos(self.angle1 + self.angle2)
        self.end_effector_y = self.length1 * np.sin(self.angle1) + self.length2 * np.sin(self.angle1 + self.angle2)

        self.last_action = np.array([0, 0])

        self.reward_history = [] 
        self.smoothing_window_size = 20 
        self.state = np.array([self.angle1, self.angle2, self.end_effector_x, self.end_effector_y], dtype=np.float32)

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed) 

        self.angle1 = np.random.uniform(0.5 * np.pi, 1.5 * np.pi)
        self.angle2 = np.random.uniform(0.5 * np.pi, 1.5 * np.pi) 

        self.angle1 = np.clip(self.angle1, 0, np.pi)
        self.angle2 = np.clip(self.angle2, 0, np.pi)

        self.prev_angle1 = self.angle1
        self.prev_angle2 = self.angle2
        self.last_action = np.array([0, 0])

        self.end_effector_x = self.length1 * np.cos(self.angle1) + self.length2 * np.cos(self.angle1 + self.angle2)
        self.end_effector_y = self.length1 * np.sin(self.angle1) + self.length2 * np.sin(self.angle1 + self.angle2)

        self.steps_done = 0 

        self.state = np.array([self.angle1, self.angle2, self.end_effector_x, self.end_effector_y], dtype=np.float32)
        
        return self.state, {}
    
    def reward_function(self):
        angle1_deviation = abs(self.angle1 - np.pi/2)
        angle1_reward = -5*np.exp(angle1_deviation) 
        
        if angle1_deviation < 0.02:
            angle1_reward += 0.3

        
        if angle1_deviation > 0.01:
            correction_reward = -0.1 * angle1_deviation
            angle1_reward += correction_reward
        else:
            correction_reward = 0
            angle1_reward += correction_reward


        combined_angle_deviation =abs((self.angle1 + self.angle2) - np.pi/2)
        combined_angle_reward = -2 * (combined_angle_deviation ** 2) 
        stability_reward=0.3 if angle1_deviation < 0.05 and combined_angle_deviation < 0.05 else 0
        action_penalty=  -0.1 * (abs(self.last_action[0]) + abs(self.last_action[1]))
        jerk_penalty = -0.3 * (abs(self.angle1 -self.prev_angle1) +abs(self.angle2 -self.prev_angle2))
        adaptive_penalty= -1.5 * (np.exp(angle1_deviation) + np.exp(combined_angle_deviation)) if angle1_deviation > 0.1 else 0
            

        if self.steps_done > 50:  
            time_penalty= -0.5 * (angle1_deviation + combined_angle_deviation)
        else:
            time_penalty=0
    
        total_reward= angle1_reward+combined_angle_reward+stability_reward+jerk_penalty+action_penalty+adaptive_penalty+time_penalty
    
        scaled_reward =total_reward/(np.pi**2)
        self.reward_history.append(scaled_reward)

        smoothed_reward = self.smooth_reward(self.reward_history, self.smoothing_window_size)
        
        return smoothed_reward
        

    def smooth_reward(self, reward_history, window_size=10):
        if len(reward_history) < window_size:
            return np.mean(reward_history)

        return np.mean(reward_history[-window_size:])

    def step(self, action):
        action = np.clip(action, -0.05, 0.05)
        
        self.angle1 += action[0]
        self.angle2 += action[1]

        self.angle1 = np.clip(self.angle1, 0, np.pi)
        self.angle2 = np.clip(self.angle2, 0, np.pi)

        self.end_effector_x = self.length1 * np.cos(self.angle1) + self.length2 * np.cos(self.angle1 + self.angle2)
        self.end_effector_y = self.length1 * np.sin(self.angle1) + self.length2 * np.sin(self.angle1 + self.angle2)

        self.state = np.array([self.angle1, self.angle2, self.end_effector_x, self.end_effector_y], dtype=np.float32)
        
        reward = self.reward_function()
        self.steps_done +=1

        terminated = False
        truncated = self.steps_done >= self.max_steps
        
        self.prev_angle1 = self.angle1
        self.prev_angle2 = self.angle2

        self.last_action = action

        info = {}

        return self.state, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError("Only human mode is supported.")

        if not hasattr(self, '_pygame_initialized'):
            pygame.init()
            self._pygame_initialized = True
        # Initialize a hidden surface to draw on
        width, height = 600, 600
        surface = pygame.Surface((width, height))

        # Set background color
        surface.fill((255, 255, 255))

        base_x, base_y = width // 2, height // 2

        joint1_x = base_x + int(self.length1 * 100 * np.cos(self.angle1))
        joint1_y = base_y - int(self.length1 * 100 * np.sin(self.angle1))

        joint2_x = joint1_x + int(self.length2 * 100 * np.cos(self.angle1 + self.angle2))
        joint2_y = joint1_y - int(self.length2 * 100 * np.sin(self.angle1 + self.angle2))

        pygame.draw.circle(surface, (0, 0, 0), (base_x, base_y), 5)
        pygame.draw.line(surface, (0, 0, 255), (base_x, base_y), (joint1_x, joint1_y), 5)
        pygame.draw.line(surface, (255, 0, 0), (joint1_x, joint1_y), (joint2_x, joint2_y), 5)
        pygame.draw.circle(surface, (0, 255, 0), (joint2_x, joint2_y), 5)

        # Convert surface to an RGB array
        frame = pygame.surfarray.array3d(surface)
        frame = np.rot90(frame)
        frame = np.rot90(frame)
        frame = np.rot90(frame)

        return frame

    def close(self):
        pygame.quit()