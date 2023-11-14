import concurrent
import time

from assistive_gym.train import render, render_pose

#### Define dynamic configs ####
# PERSON_IDS = ['p001', 'p002', 'p003', 'p004', 'p005', 'p006', 'p007', 'p008', 'p009', 'p010', 'p011', 'p012',
#               'p013', 'p014', 'p015', 'p016', 'p017', 'p018', 'p019', 'p020', 'p021', 'p022', 'p023', 'p024', 'p025',
#               'p026', 'p027', 'p028', 'p029', 'p030', 'p031', 'p032', 'p033', 'p034', 'p035', 'p036', 'p037', 'p038',
#               'p039', 'p040', 'p041', 'p042', 'p043', 'p044', 'p045']
# # PERSON_IDS = ['p004']
# # SMPL_FILES = ['s01' ]
# SMPL_FILES = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11', 's12',
#               's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26',
#               's27', 's28', 's29', 's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's40',
#               's41', 's42', 's43', 's44', 's45']

PERSON_IDS = ['p001']
SMPL_FILES = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11', 's12',]
# SMPL_FILES = [ 'S01']
# SMPL_FILES = [ 's44']
# PERSON_IDS = [ 'p001', 'p002']
# SMPL_FILES = [ 's19', 's20', 's44', 's45']
OBJECTS = ['cane']
#### Define static configs ####
SMPL_DIR = 'examples/data/slp3d/'
ENV = "HumanComfort-v1"
SEED = 1001
SAVE_DIR = 'trained_models'
SIMULATE_COLLISION = False
ROBOT_IK = True
END_EFFECTOR = 'right_hand'

### DEFINE MULTIPROCESS SETTING ###
NUM_WORKERS = 1

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
            # render_pose(ENV, p, s)
        do_render(config)

    # TODO: bring back
    #
    # with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    #     results = executor.map(do_render, configs)
    # for num, result in enumerate(results):
    #     print('Done training for {} {}'.format(num, result))
