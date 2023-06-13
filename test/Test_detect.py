import numpy as np
from sklearn.preprocessing import StandardScaler

from detect import compute_canopy_height_model, gaussian_chm, detect_trees, compute_original_coordinates, \
    detect_tubular_form2, plot_simulated_trees_3d
from utils import load_las


def test_detection():
    point_cloud = load_las(r"C:\Users\Xiao\PycharmProjects\pythonProject\part_3.las")

    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(point_cloud)
    points_scaled2 = points_scaled

    resolution = 0.005  # meters

    # perform chm algorithm
    chm = compute_canopy_height_model(points_scaled2, resolution)
    chm = gaussian_chm(chm, 2)

    threshold = 0.3  # Adjust this value to control tree detection sensitivity
    filter_sizer = 120  # Adjust this value to control tree detection sensitivity

    tree_indices = detect_trees(chm, threshold, filter_sizer)

    max_coords = np.max(points_scaled, axis=0)

    tree_indices_2 = tree_indices

    original_coords = compute_original_coordinates(tree_indices_2, resolution, max_coords, chm)
    original_coords_2 = scaler.inverse_transform(original_coords)

    detection2 = detect_tubular_form2(point_cloud, original_coords_2, 2)
    plot_simulated_trees_3d(detection2, point_cloud)
    assert len(detection2) == 11
