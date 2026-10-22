import gymnasium as gym
from stable_baselines3 import PPO

model = PPO.load("ppo_cartpole_v1")

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
