import time

from assistive_gym.train import train

#### Define dynamic configs ####
# PERSON_IDS = ['p001', 'p002', 'p003', 'p004', 'p005']
# SMPL_FILES = ['s01', 's02', 's03', 's04', 's05']
# SMPL_FILES = ['s06', 's07', 's08', 's09', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19']
PERSON_IDS = ['p001', 'p002', 'p003', 'p004', 'p005']
SMPL_FILES = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20']
OBJECTS = ['cane', 'cup', 'pill']

#### Define static configs ####
SMPL_DIR = 'examples/data/slp3d/'
ENV = 'HumanComfort-v1'
SEED = 1001
SAVE_DIR = 'trained_models'
RENDER_GUI = False
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

import concurrent.futures

def do_train(config):
    p, s, o = config
    print (p, s, o)
    train(ENV, SEED, s, p, END_EFFECTOR,  SAVE_DIR, RENDER_GUI, SIMULATE_COLLISION, ROBOT_IK, o)
    return "Done training for {} {} {}".format(p, s, o)


if __name__ == '__main__':
    configs = get_dynamic_configs()

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = executor.map(do_train, configs)

    print('Results:')
    for num, result in enumerate(results):
        print('The square of {} is {}'.format(num, result))
