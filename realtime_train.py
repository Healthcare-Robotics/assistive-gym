import argparse
import json
import os
import shutil

import pybullet_data
import pybullet as p

from assistive_gym.envs.utils.urdf_utils import generate_human_mesh
from assistive_gym.mprocess_train import mp_train
from assistive_gym.train import train, render
from experimental.urdf_name_resolver import get_urdf_folderpath, get_urdf_ref_filepath

def generate_urdf(args):
    # Start the simulation engine
    physic_client_id = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    urdf_folder = get_urdf_folderpath(args.person_id)
    os.makedirs(urdf_folder, exist_ok=True)

    urdf_ref_file = get_urdf_ref_filepath(urdf_folder)
    shutil.copy(args.ref_urdf_file , urdf_ref_file)

    generate_human_mesh(physic_client_id, args.gender, urdf_ref_file, urdf_folder, args.smpl_file)

    # remove the ref file from the folder
    os.remove(urdf_ref_file)
    # Disconnect from the simulation
    p.disconnect()

def check_urdf_exist(person_id):
    urdf_folder = get_urdf_folderpath(person_id)
    # check if file exists
    urdf_file = os.path.join(urdf_folder, 'human.urdf')
    return os.path.exists(urdf_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Util script for human urdf generation and training')
    parser.add_argument('--ref-urdf-file', default='assistive_gym/envs/assets/human/ref_mesh.urdf',  help='Path to reference urdf file')
    parser.add_argument('--person-id', default='mane', help='Person id')
    parser.add_argument('--smpl-file', default='/nethome/nnagarathinam6/Documents/joint_reaching_evaluation/BodyPressureTRI/networksscan7.pkl',
                        help='Path to smpl file')
    parser.add_argument('--gender', default='male', help='Gender')
    parser.add_argument('--end-effector', default='right_hand', help='End effector')
    parser.add_argument('--handover-object', default='pill', help='Handover object')
    args = parser.parse_args()
    # if not check_urdf_exist(args.person_id):
    #     generate_urdf(args)
    generate_urdf(args)
    result = mp_train('HumanComfort-v1', 1001, args.smpl_file, args.person_id, args.end_effector,
          'trained_models', True, False, True, args.handover_object)
    # print(realworld['wrt_pelvis'])

    # render('HumanComfort-v1', args.person_id, args.smpl_file, 'trained_models', args.handover_object, True)
