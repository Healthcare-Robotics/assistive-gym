from assistive_gym.envs.utils.urdf_utils import *

smpl_path = 'examples/data/p001/s01.pkl'
smpl_data = load_smpl(smpl_path)


print (smpl_data.body_pose.shape)