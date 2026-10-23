import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class AppleCatcherEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        
        self.width = 600
        self.height = 600
        self.render_mode = render_mode
        
        # 3 actions: 0 = Move Left, 1 = Stay, 2 = Move Right
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [basket_x, apple_x, apple_y] normalized from 0 to 1
        self.observation_space = spaces.Box( low=0.0, high=1.0, shape=(3,), dtype=np.float32 )
        
        self.basket_x = 0.5
        self.basket_width = 0.2
        self.apple_x = 0.5
        self.apple_y = 0.0
        self.apple_speed = 0.03
        
        self.window = None
        self.clock = None
        if self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Apple Catcher RL")
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.basket_x = 0.5
        self.apple_x = float(self.np_random.uniform(0.1, 0.9))
        self.apple_y = 0.0
        
        observation = np.array([self.basket_x, self.apple_x, self.apple_y], dtype=np.float32)
        return observation, {}

    def step(self, action):
        if action == 0:
            self.basket_x = max(0.0, self.basket_x - 0.05)
        elif action == 2:
            self.basket_x = min(1.0, self.basket_x + 0.05)
            
        self.apple_y += self.apple_speed
        
        reward = 0.0
        terminated = False
        truncated = False
        
        if self.apple_y >= 0.95:
            basket_left = self.basket_x - (self.basket_width / 2)
            basket_right = self.basket_x + (self.basket_width / 2)
            
            if basket_left <= self.apple_x <= basket_right:
                reward = 1.0  # Caught!
            else:
                reward = -1.0 # Missed!
                
            terminated = True # Episode ends when apple hits the ground

        observation = np.array([self.basket_x, self.apple_x, self.apple_y], dtype=np.float32)
        
        if self.render_mode == "human":
            self.render()
            
        return observation, reward, terminated, truncated, {}

    def render(self):
        if self.window is None:
            return
            
        self.window.fill((255, 255, 255)) # White background
        
        # Basket 
        bx_px = int((self.basket_x - self.basket_width / 2) * self.width)
        bw_px = int(self.basket_width * self.width)
        pygame.draw.rect(self.window, (0, 0, 255), (bx_px, self.height - 40, bw_px, 20))
        
        # Apple
        ax_px = int(self.apple_x * self.width)
        ay_px = int(self.apple_y * self.height)
        pygame.draw.circle(self.window, (255, 0, 0), (ax_px, ay_px), 10)
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()

env = AppleCatcherEnv(render_mode="human")
obs, info = env.reset()

for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
