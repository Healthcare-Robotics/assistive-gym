import json

from assistive_gym.envs.utils.urdf_utils import *

SMPL_SOURCEDIR = 'examples/data/slp3d'
SMPL_OUTDIR = 'deepnn/data/input/searchinput'

# looop through all the folders
for folder in os.listdir(SMPL_SOURCEDIR):
    dir = os.path.join(SMPL_SOURCEDIR, folder)
    if os.path.isdir(dir):
        for file in os.listdir(dir):
            if file.endswith(".pkl"):
                smpl_path = os.path.join(dir, file)
                smpl_data = load_smpl(smpl_path)
                # output smpl_data to json
                json_path = os.path.join(SMPL_OUTDIR, folder, file.replace('.pkl', '.json'))
                # make sure the directory exists
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                with open(json_path, 'w') as f:
                    # build json
                    data = {}
                    data['pose'] = smpl_data.body_pose.tolist()
                    # put back the global orientation to the first 3 elements
                    data['pose'][0:3] = smpl_data.global_orient.tolist()
                    data['betas'] = smpl_data.betas.tolist()
                    data['transl'] = smpl_data.transl.tolist()

                    f.write(json.dumps(data, indent=4))
