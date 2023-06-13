import time
import csv
import open3d as o3d
import numpy as np
import pylas
import laspy



def measure_execution_time(func):

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"The function {func.__name__} took {execution_time:.6f} seconds to execute.")

        with open(f'csv/{func.__name__}-execution_times.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([execution_time])

        return result

    return wrapper


def display_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.visualization.draw_geometries([pcd])


def load_las(file_path):
    # load laz file
    las = laspy.read(file_path)
    # get points
    points = np.vstack((las.x, las.y, las.z)).transpose()

    return points


def save_las(points, save_path):
    las_file = pylas.create()

    # add points to las file
    las_file.x = points[:, 0]
    las_file.y = points[:, 1]
    las_file.z = points[:, 2]

    # save las file
    las_file.write(save_path)