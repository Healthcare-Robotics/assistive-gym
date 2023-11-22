import argparse

from assistive_gym.envs.utils.train_utils import render

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Util script for human urdf generation and training')
    parser.add_argument('--ref-urdf-file', default='assistive_gym/envs/assets/human/ref_mesh.urdf',  help='Path to reference urdf file')
    parser.add_argument('--person-id', default='matt', help='Person id')
    parser.add_argument('--smpl-file', default='/nethome/nnagarathinam6/Documents/joint_reaching_evaluation/BodyPressureTRI/networks/scan/s3.pkl',
                        help='Path to smpl file')
    parser.add_argument('--gender', default='male', help='Gender')
    parser.add_argument('--end-effector', default='right_hand', help='End effector')
    parser.add_argument('--handover-object', default='cane', help='Handover object')
    args = parser.parse_args()
    # if not check_urdf_exist(args.person_id):
    #     generate_urdf(args)

    render('HumanComfort-v1', args.person_id, args.smpl_file, 'trained_models', args.handover_object, True)