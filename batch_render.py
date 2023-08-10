import time

from assistive_gym.train import render, render_pose

#### Define dynamic configs ####
PERSON_IDS = [ 'p002']
# SMPL_FILES = ['s05']
# PERSON_IDS = ['p002']
# SMPL_FILES = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20']
SMPL_FILES = ['s01']
OBJECTS = ['cane']

#### Define static configs ####
SMPL_DIR = 'examples/data/slp3d/'
ENV = 'HumanComfort-v1'
SEED = 1001
SAVE_DIR = 'trained_models'
SIMULATE_COLLISION = False
ROBOT_IK = True
END_EFFECTOR = 'right_hand'

### DEFINE MULTIPROCESS SETTING ###
NUM_WORKERS = 20

def get_dynamic_configs():
    configs =[]
    for p in PERSON_IDS:
        for s in SMPL_FILES:

            for o in OBJECTS:
                smpl_file = SMPL_DIR + p + '/' + s + '.pkl'
                # print(p, s, o)
                configs.append((p, smpl_file, o))
    return configs

def do_render(config):
    p, s, o = config
    print (p, s, o)
    render(ENV, p, s, SAVE_DIR,o, ROBOT_IK)

if __name__ == '__main__':
    configs = get_dynamic_configs()

    displayed = set()
    for config in configs:
        p, s, o = config
        if (p, s) not in displayed:
            displayed.add((p, s))
            render_pose(ENV, p, s)
        do_render(config)
