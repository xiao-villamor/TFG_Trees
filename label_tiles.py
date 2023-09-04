import os

import laspy
import numpy as np
import open3d as o3d

def mark_points(file_path):
    # Read the LAS/LAZ file

    print(file_path)

    path = 'D:\Data\Datatest\Data 2\LIDAR2019_NdP_81500_82500_EPSG2169\\no_ground\\' + file_path
    las = laspy.read(path)
    point_cloud = np.vstack((las.x, las.y, las.z)).transpose()

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Create Open3D visualizer
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    points = vis.get_picked_points()

    points = np.array(pcd.points)[points]

    # Save the points in a csv file with the name of the las file
    np.savetxt("D:\Data\Datatest\Data 2\LIDAR2019_NdP_81500_82500_EPSG2169\groundT\\" + file_path + '.csv', points, delimiter=",")


if __name__ == '__main__':
    # Open cloud of points with ground



    # get the file path of the las file in the extracted folder

    files = os.listdir(f'D:\Data\Datatest\Data 2\LIDAR2019_NdP_81500_82500_EPSG2169\\no_ground\\')

    for file in files:
        mark_points(file)

