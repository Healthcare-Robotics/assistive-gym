from gym.envs.registration import register

tasks = ['ScratchItch', 'BedBathing', 'Feeding', 'Drinking', 'Dressing', 'ArmManipulation']
robots = ['PR2', 'Jaco', 'Baxter', 'Sawyer', 'Stretch']
human_states = ['', 'Human']

for task in tasks:
    for robot in robots:
        for human_state in human_states:
            register(
                id='%s%s%s-v1' % (task, robot, human_state),
                entry_point='assistive_gym.envs:%s%s%sEnv' % (task, robot, human_state),
                max_episode_steps=200,
            )

for task in ['ScratchItch']:
    for robot in robots:
        register(
            id='%s%sMesh-v1' % (task, robot),
            entry_point='assistive_gym.envs:%s%sMeshEnv' % (task, robot),
            max_episode_steps=200,
        )

register(
    id='HumanTesting-v1',
    entry_point='assistive_gym.envs:HumanTestingEnv',
    max_episode_steps=200,
)

register(
    id='SMPLXTesting-v1',
    entry_point='assistive_gym.envs:SMPLXTestingEnv',
    max_episode_steps=200,
)

