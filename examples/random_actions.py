import assistive_gym
import gymnasium as gym

env = gym.make('FeedingPR2-v0')
env.render()
observation, _ = env.reset()

while True:
    env.render()
    action = env.action_space.sample() # Get a random action
    observation, reward, done, truncated, info = env.step(action)
