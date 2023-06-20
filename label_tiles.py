import os

import laspy
import numpy as np
import open3d as o3d
import requests as requests
import zipfile as zipfile


def mark_points(file_path):
    # Read the LAS/LAZ file

    path = 'data\Label_data\\' + file_path
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
    np.savetxt(file_path + '.csv', points, delimiter=",")

    # delete the las file
    os.remove(path)


if __name__ == '__main__':
    # Open cloud of points with ground

    url = "https://files.capibara.dev/api/public/dl/w10rbCFg"

    # Download the file
    r = requests.get(url)
    # Save the file in R

    open('data.zip', 'wb').write(r.content)

    # Extract the file
    with zipfile.ZipFile('data.zip', 'r') as zip_ref:
        zip_ref.extractall('data')

    # get the file path of the las file in the extracted folder

    files = os.listdir('data\Label_data')

    for file in files:
        mark_points(file)

    # delete the zip file
    os.remove('data.zip')

