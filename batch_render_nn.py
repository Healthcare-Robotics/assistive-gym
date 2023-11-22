import json
import os

from assistive_gym.envs.utils.train_utils import get_save_dir, render_nn_result

#### Define dynamic configs ####
PERSON_IDS = ['p001']
SMPL_FILES = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11', 's12',]

OBJECTS = ['cane']
#### Define static configs ####
SMPL_DIR = 'examples/data/slp3d/'
ENV = "HumanComfort-v1"
# SAVE_DIR = 'trained_models'
SAVE_DIR = 'deepnn/data/output'

def get_dynamic_configs():
    configs = []
    for p in PERSON_IDS:
        for s in SMPL_FILES:
            for o in OBJECTS:
                smpl_file = SMPL_DIR + p + '/' + s + '.pkl'
                # print(p, s, o)
                configs.append((p, smpl_file, o))
    return configs


def do_render_nn(config):
    p, s, o = config
    print(p, s, o)
    save_dir = get_save_dir(SAVE_DIR, "", p, s)
    # load data from json file. filename = object name, file dir = save_dir/p/s
    data = json.load(open(os.path.join(save_dir, o + ".json"), "r"))
    data['end_effector'] ='right_hand'
    render_nn_result(ENV, data, p, s, o)


if __name__ == '__main__':
    configs = get_dynamic_configs()
    for config in configs:
        do_render_nn(config)
