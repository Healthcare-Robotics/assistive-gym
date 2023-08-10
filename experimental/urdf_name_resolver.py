import os

BASEDIR = os.path.dirname(os.path.realpath(__file__))

URDF_FOLDER = 'urdf'
def get_urdf_folderpath(person_id):
    return os.path.join(BASEDIR, URDF_FOLDER, person_id)


def get_urdf_filepath(urdf_folder):
    return os.path.join(urdf_folder, 'human.urdf')


def get_urdf_ref_filepath(urdf_folder):
    return os.path.join(urdf_folder, 'ref_mesh.urdf')


def get_urdf_mesh_folderpath(urdf_folder):
    return os.path.join(urdf_folder, 'meshes')
