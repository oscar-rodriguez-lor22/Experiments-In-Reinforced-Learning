import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1)

print("Training started...")
model.learn(total_timesteps=10000)

model.save("ppo_cartpole_v1")
print("Model trained and saved as 'ppo_cartpole_v1.zip'!")
