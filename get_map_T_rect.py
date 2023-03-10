import open3d as o3d
import sys
import mrob
import numpy as np
import os, glob, copy
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import argparse


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--root_folder",
    type=str,
    default='/home/anastasia/personal/mnt/patata/datasets/navigine/mocab/apriltag/Azure/mocap',
    help="root directory of the dataset where the depth and rgb images are located",
)

def run_FGraph(x):

    # How to align a vector with gravity, ensuring that only the rotation is changed
    # For this we need two factors, one looking at the rotation and the second looking at the pose to remain at the origin.

    # Current vector, i
    # x = np.random.randn(3) # Inpout max vector from PCA
    x = x / np.linalg.norm(x)

    # Desired vector, in this case Z axis
    y = np.zeros(3)
    y[2] = 1

    # Solve for one point, it is rank defincient, but the solution should be the geodesic (shortest) path
    graph = mrob.FGraph()
    W = np.eye(3)
    n1 = graph.add_node_pose_3d(mrob.geometry.SE3())
    graph.add_factor_1pose_point2point(z_point_x = x,
                                        z_point_y = y,
                                        nodePoseId = n1,
                                        obsInf = W)

    # Add anchor factor for position
    W_0 = np.identity(6)*1e6
    W_0[:3,:3] = np.identity(3)*1e-4
    graph.add_factor_1pose_3d(mrob.geometry.SE3(),n1,W_0)

    graph.solve(mrob.LM)
    T = graph.get_estimated_state()
    # print(T)
    # graph.print(True)

    return T[0]

def read_associations(root_folder, idx):
    with open(os.path.join(root_folder,'associations.txt')) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    associations = []
    
    for i in lines:
        associations.append(i.split(" ")[1::2])
    return associations[idx]

def quat_to_SE3(quat_pose):
    
    rot_4x1 = quat_pose[-4:]
    tra_3x1 = quat_pose[1:4]

    rot = mrob.geometry.quat_to_so3(rot_4x1)
    pose = mrob.geometry.SE3(mrob.geometry.SO3(rot),tra_3x1)

    return pose


def get_3d_pose(pose, size = 0.1):
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    return axis.transform(pose)

def get_data(root_folder, idx):

    pairs = read_associations(root_folder,idx)
    color_dir = root_folder + pairs[0]
    depth_dir = root_folder + pairs[1]
    
    color = o3d.t.io.read_image(color_dir)
    depth = o3d.t.io.read_image(depth_dir)
    return color, depth

def get_pcd(root_folder, camera_matrix, idx):

    color, depth = get_data(root_folder, idx)

    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.intrinsic_matrix = camera_matrix

    rgbd = o3d.t.geometry.RGBDImage(color, depth)
                                   
    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_matrix, depth_scale=1000.0, depth_max=3.0)
    pcd = pcd.to_legacy()

    return pcd



if __name__ == '__main__':
    args = parser.parse_args()   

    camera_intrinsic = np.array([[953.95397949, 0, 958.03153013], [0, 941.55212402, 552.51219511], [0, 0, 1]])

    orb_poses = pd.read_csv(os.path.join(args.root_folder,'CameraTrajectory.txt'), header=None, sep="\s+")
    orb_quat = np.array([each for _,each in orb_poses.iterrows()])


    orb = np.zeros((orb_quat.shape[0],4,4))

    for i in range(len(orb_quat)):
        orb[i,:,:] = (quat_to_SE3(orb_quat[i,:])).T()

    pcd_globs = []
    start, end, step = 0, orb.shape[0], 50

    for idx in tqdm(range(start, end, step)):
        pcd = get_pcd(args.root_folder, camera_intrinsic, idx)
        pcd_globs.append(pcd.transform(orb[idx,:,:]).voxel_down_sample(voxel_size=0.05))


    pcd_all = o3d.geometry.PointCloud()
    for idx, pcd in enumerate(tqdm(pcd_globs)):
        pcd_all += pcd
    del pcd_globs

    cl, ind = pcd_all.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    map_rect_filtered = pcd_all.select_by_index(ind)
    del pcd_all

    pca = PCA(n_components=3)
    pca.fit(map_rect_filtered.points)

    T_graph = run_FGraph(pca.components_[2])
    map_rect = copy.deepcopy(map_rect_filtered).transform(T_graph)


    points = np.asarray(map_rect.points)
    colors = np.asarray(map_rect.colors)
    _,_,z = np.asarray(map_rect.points).T



    Points_ = list()
    Colors_ = list()
    print('cropping points too keep only wall s')
    for z_, point, color in zip(tqdm(z), points, colors):
        ## specify z-axis threshold to remove top and down points 
        if z_ < 0.8 * z.max() and z_ > 0.2 * z.min():
            Points_.append(point)
            Colors_.append(color)

    points_ = np.asarray(Points_)
    colors_ = np.asarray(Colors_)

    aligned_map_rect = o3d.geometry.PointCloud()
    aligned_map_rect.points = o3d.utility.Vector3dVector(points_)
    aligned_map_rect.colors = o3d.utility.Vector3dVector(colors_)

    o3d.visualization.draw_geometries([aligned_map_rect], window_name='aligned_map_rect')

    print(f'writng to: {args.root_folder}')

    # save_map_to = os.path.join(args.root_folder,'map_rect_walls.pcd')
    # save_T_to = os.path.join(args.root_folder,'T_rect.npy')
    save_map_to = os.path.join('map_rect_walls.pcd')
    save_T_to = os.path.join('T_rect.npy')

    o3d.io.write_point_cloud(save_map_to, aligned_map_rect)
    np.save(save_T_to, T_graph)