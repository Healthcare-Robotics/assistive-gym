import assistive_gym
import gymnasium as gym

env = gym.make('FeedingPR2-v0')
env.render()
observation = env.reset()

while True:
    env.render()
    action = env.action_space.sample() # Get a random action
    observation, reward, done, info = env.step(action)
