import numpy as np
from sklearn.preprocessing import StandardScaler

from visualization_utils import plot_tree_locations
from detect_utils import check_accuracy
from detect import compute_canopy_height_model, gaussian_chm, detect_trees, compute_original_coordinates, \
    detect_tubular_form2

from utils import load_las


def test_detection():
    point_cloud = load_las(r"C:\Users\Xiao\PycharmProjects\pythonProject\part_2.las")

    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(point_cloud)
    points_scaled2 = points_scaled

    resolution = 0.005  # meters

    # perform chm algorithm
    chm = compute_canopy_height_model(points_scaled2, resolution)
    chm = gaussian_chm(chm, 2)

    threshold = 0.3  # Adjust this value to control tree detection sensitivity
    filter_sizer = 105  # Adjust this value to control tree detection sensitivity

    tree_indices = detect_trees(chm, threshold, filter_sizer)

    max_coords = np.max(points_scaled, axis=0)

    tree_indices_2 = tree_indices

    original_coords = compute_original_coordinates(tree_indices_2, resolution, max_coords, chm)
    original_coords_2 = scaler.inverse_transform(original_coords)

    # plot_tree_locations(tree_indices, chm_or)
    plot_tree_locations(tree_indices, chm)
    # Open cloud of points without ground
    file_path_no_ground = r"C:\Users\Xiao\PycharmProjects\pythonProject\part_2_no_ground.las"
    # Read the LAS/LAZ file
    point_cloud_no_ground = load_las(file_path_no_ground)

    detection2 = detect_tubular_form2(point_cloud_no_ground, original_coords_2, 3)
    # display_original_cloud_with_dot(point_cloud_no_ground, tree_indices ,"purple")
    # plot_simulated_trees_o3d(detection2, point_cloud_no_ground)
    print("accuracy: " + str(check_accuracy(detection2) * 100) + "%")

    assert len(detection2) == 12
