import os
import numpy as np
import smplx
import trimesh
import torch
from assistive_gym.envs.utils.smpl_parser import SMPL_Parser
import open3d as o3d

class HullWrapper:
    def __init__(self, hull: trimesh.Trimesh, filename: str):
        self.hull = hull
        self.filename = filename


# BETAS = torch.Tensor(np.random.uniform(-1, 5, (1, 10))) # random betas
BETAS = torch.Tensor(np.zeros((1, 10)))  # default beta
DENSITY = 1000


def generate_body_hull(jname, vert, outdir, joint_pos=(0, 0, 0)):
    """
    Generate a convex hull for each joint
    Save the convex hull as an obj file with vert name
    :param jname: joint name
    :param vert: array of vertices representing the joint
    :return:
    """

    p_cloud = trimesh.PointCloud(vert)
    p_hull = p_cloud.convex_hull
    # p_hull = simplify_mesh(p_hull, 4)
    # p_hull = smooth_mesh(p_hull)
    # p_hull.show()
    p_hull.density = DENSITY
    centroid = p_hull.centroid

    # the original hull will render at exactly where the body part should be.
    # we move the body part to origin so that we could put it to the right place with joint position later
    p_hull.vertices = p_hull.vertices - joint_pos

    print(jname, "pos: ", joint_pos, " centroid: ", centroid, " volume: ", p_hull.volume, " area: ", p_hull.area,
          " inertia: ", p_hull.moment_inertia, )
    # Export the mesh to an OBJ file
    outfile = f"{outdir}/{jname}.obj"
    p_hull.export(outfile)
    return {
        "filename": outfile,
        "hull": p_hull,
    }


def generate_geom(default_model_path, smpl_data= None, outdir=None):
    smpl_parser = SMPL_Parser(default_model_path)
    pose = torch.zeros((1, 72)) # reset the model to default pose

    # pose = torch.Tensor(smpl_data.body_pose).unsqueeze(0)
    # print ("pose shape: ", pose.shape)
    transl = None
    if smpl_data is not None:
        print ("betas before: ", smpl_data.betas)
        betas = torch.Tensor(np.array(smpl_data.betas).reshape(1, 10))
        if smpl_data.transl is not None:
            transl = torch.Tensor(smpl_data.transl).unsqueeze(0)
    else:
        betas = BETAS

    (
        smpl_verts,
        smpl_jts,
        skin_weights,
        joint_names,
        joint_offsets,
        joint_parents,
        joint_axes,
        joint_dofs,
        joint_range,
        contype,
        conaffinity,
    ) = smpl_parser.get_mesh_offsets(pose, betas=betas, transl=transl)

    vert_to_joint = skin_weights.argmax(axis=1)
    hull_dict = {}
    scale_dict = {
        # "Spine3": 0.9,
    }
    # create joint geometries
    # print("need to change geom_dir in smpl_geom.py line 79")
    # geom_dir = "/home/hrl5/assistive-gym/assistive_gym/envs/assets/human/meshes/"
    # geom_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../assets/human/meshes/")
    os.makedirs(outdir, exist_ok=True)
    joint_pos_dict = {}
    total_mass = 0
    for jind, jname in enumerate(joint_names):
        vind = np.where(vert_to_joint == jind)[0]
        if len(vind) == 0:
            print(f"{jname} has no vertices!")
            continue
        vert = (smpl_verts[vind] - smpl_jts[jind]) * scale_dict.get(jname, 1) + smpl_jts[jind]
        # vert = (smpl_verts[vind] - smpl_jts[jind]) + smpl_jts[jind]
        r = generate_body_hull(jname, vert, outdir, joint_pos=smpl_jts[jind])
        joint_pos_dict[jname] = smpl_jts[jind]

        hull_dict[jname] = HullWrapper(r["hull"], r["filename"])
        total_mass += r["hull"].mass
    print("total mass: ", total_mass)
    return hull_dict, joint_pos_dict, joint_offsets


def simplify_mesh(mesh, reduce_factor=2):
    """
    Simplify the mesh by quadratic decimation
    :param mesh:
    :param reduce_factor:
    :return:
    """
    target_face_count = len(mesh.faces) // reduce_factor  # Reduce face count by 50%
    new_mesh = mesh.simplify_quadratic_decimation(target_face_count)
    return new_mesh


def smooth_mesh(mesh: trimesh.Trimesh):
    """
    Smooth the mesh by laplacian smoothing
    :param mesh:
    :return:
    """
    mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=10)
    return mesh


def visualize_pointcloud(vert):
    """
    visualize a point cloud from vertices of a body part for debugging
    :param vert: vertices of a mesh
    :return:
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vert)
    o3d.visualization.draw_geometries([pcd])


def show_human_mesh(model_path):
    """
    Show the human mesh for debugging
    :param model_path:
    :return:
    """
    model = smplx.create(model_path, model_type='smpl', gender='neutral')
    output = model(betas=BETAS, body_pose=torch.Tensor(np.zeros((1, model.NUM_BODY_JOINTS * 3))), return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    out_mesh = trimesh.Trimesh(vertices, model.faces)
    print ("volume: ", out_mesh.volume, " area: ", out_mesh.area, " inertia: ", out_mesh.moment_inertia,)
    out_mesh.show()


if __name__ == "__main__":
    model_path = os.path.join(os.getcwd(), "examples/data/SMPL_NEUTRAL.pkl")

    generate_geom(model_path)
    show_human_mesh(model_path)
