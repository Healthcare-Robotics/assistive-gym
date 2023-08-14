import argparse

import pybullet as p
import pybullet_data

from assistive_gym.envs.utils.urdf_utils import generate_human_mesh


def generate_urdf(args):
    # Start the simulation engine
    physic_client_id = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    generate_human_mesh(physic_client_id, args.ref_urdf_file, args.out_urdf_file, args.smpl_file)
    print(f"Generated urdf file from ref urdf: { args.ref_urdf_file} with smpl file: {args.smpl_file}, out file: {args.out_urdf_file}")

    # Disconnect from the simulation
    p.disconnect()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Util script for human urdf generation')
    parser.add_argument('--ref-urdf-file', default='ref_mesh.urdf',  help='Path to output urdf file')
    parser.add_argument('--out-urdf-file', default='test_mesh.urdf',
                        help='Path to output urdf file')
    parser.add_argument('--smpl-file', default='examples/data/smpl_bp_ros_smpl_re2.pkl',
                        help='Path to smpl file')
    parser.add_argument('--mode', default='generate', help='Mode: generate or test')
    args = parser.parse_args()
    if args.mode == 'generate':
        generate_urdf(args)
    else:
        raise NotImplementedError()

