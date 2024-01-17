import assistive_gym
try:
    import gymnasium as gym
except ImportError:
    import gym

env = gym.make('FeedingPR2-v0')
observation, _ = env.reset()
env.render()

while True:
    env.render()
    action = env.action_space.sample() # Get a random action
    observation, reward, done, truncated, info = env.step(action)
