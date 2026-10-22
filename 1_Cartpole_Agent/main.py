import gymnasium as env_gym

# Create the CartPole environment with visual rendering
env = env_gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
