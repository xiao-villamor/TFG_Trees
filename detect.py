import laspy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import maximum_filter, gaussian_filter, median_filter, uniform_filter
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler
from skimage.restoration import denoise_bilateral

import open3d as o3d

from utils import measure_execution_time


def gaussian_chm(chm_gaussian, sigma):
    # Apply Gaussian smoothing to the CHM
    filtered_chm = gaussian_filter(chm_gaussian, sigma=sigma)

    return filtered_chm


def median_filter_chm(chm_median, size):
    # Apply median filtering to the CHM
    filtered_chm = median_filter(chm_median, size=size)

    return filtered_chm


def mean_filter_chm(chm_mean, size):
    # Apply mean filtering to the CHM
    filtered_chm = uniform_filter(chm_mean, size=size)

    return filtered_chm


def bilateral_filter_chm(chm_bilateral, sigma_spatial, sigma_range):
    # Apply bilateral filtering to the CHM
    filtered_chm = denoise_bilateral(chm_bilateral, sigma_color=sigma_range, sigma_spatial=sigma_spatial)

    return filtered_chm


def estimate_ground_surface(point_cloud, ransac_threshold=0.05, ransac_iterations=100):
    # Separate x, y, and z coordinates
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    # Create a RANSAC regressor object
    ransac = RANSACRegressor(min_samples=5, residual_threshold=ransac_threshold, max_trials=ransac_iterations)

    # Fit the RANSAC model to the data
    ransac.fit(np.column_stack((x, y)), z)

    # Get inlier indices (representing ground points)
    inlier_mask = ransac.inlier_mask_

    # Estimate the ground surface by using the maximum z value of the inlier points
    ground_surface = np.max(z[inlier_mask])

    return ground_surface


def compute_canopy_height_model(point_cloud_height_model, resolution):
    min_coords_height_model = np.min(point_cloud_height_model, axis=0)
    max_coords_height_model = np.max(point_cloud_height_model, axis=0)

    # Create a regular grid based on the resolution
    x_grid = np.arange(min_coords_height_model[0], max_coords_height_model[0], resolution)
    y_grid = np.arange(min_coords_height_model[1], max_coords_height_model[1], resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)

    # Flatten the grid coordinates
    xy_flat = np.column_stack((xx.ravel(), yy.ravel()))

    # Estimate the ground surface using RANSAC
    ground_surface = estimate_ground_surface(point_cloud_height_model)

    # Build a KD-tree from the point cloud
    kdtree = cKDTree(point_cloud_height_model[:, :2])

    # Query the nearest neighbors for each grid point
    _, indices = kdtree.query(xy_flat, k=1)

    # Compute the canopy height for each grid point relative to the ground
    canopy_height = point_cloud_height_model[indices, 2]

    # Reshape the canopy height to match the grid shape
    canopy_height_model = canopy_height.reshape(xx.shape)
    return canopy_height_model


def display_canopy_height_model(canopy_height_model, resolution):
    # Create x and y coordinates for the grid
    x_coords = np.arange(0, canopy_height_model.shape[1]) * resolution
    y_coords = np.arange(0, canopy_height_model.shape[0]) * resolution

    # Create a meshgrid from the coordinates
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Plot the canopy height model
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, canopy_height_model, cmap='viridis')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Canopy Height')
    ax.set_title('Canopy Height Model')

    # Show the plot
    plt.show()


@measure_execution_time
def detect_trees(canopy_height_model, threshold, filter_size):
    # Apply local maximum filtering to find peaks
    neighborhood_size = (filter_size, filter_size)
    local_max = maximum_filter(canopy_height_model, footprint=np.ones(neighborhood_size), mode='constant')
    tree_mask = (canopy_height_model == local_max) & (canopy_height_model > threshold)

    # Find the indices of the tree locations
    tree_indices = np.where(tree_mask)

    return tree_indices


def plot_point_cloud_2d(point_cloud):
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    # rotate the point cloud

    plt.scatter(x, y, c=z, cmap='viridis', s=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Point Cloud with Height')
    plt.colorbar(label='Height')
    plt.show()


def plot_point_cloud(point_cloud, tree_indices):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, and z coordinates from the point cloud
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    # Plot the points
    ax.scatter(x, y, z, c='b', marker='.', s=1)

    # Plot the trees
    for tree_index in tree_indices:
        ax.scatter(x[tree_index], y[tree_index], z[tree_index], c='r', marker='.', s=1)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud')

    # Show the plot
    plt.show()


def display_original_cloud_with_dot(point_cloud, original_coords):
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    geometries = [pcd]

    # Create a red dot for each tree location
    for coords in original_coords:
        dot = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
        # color the dot purple
        dot.paint_uniform_color([0.5, 0, 0.5])

        # Translate the dot to the tree location respective to the origin 0, 0, 0
        dot.translate(coords)

        geometries.append(dot)

    # crete a perpendicular line to the ground for each tree starting at the tree location minus 10 meters
    for coords in original_coords:
        line = o3d.geometry.LineSet.create_from_triangle_mesh(
            o3d.geometry.TriangleMesh.create_cylinder(radius=0.1, height=80))
        line.paint_uniform_color([0, 0, 0.5])
        line.translate(coords)
        geometries.append(line)

    # Create an Open3D visualization window and add geometries
    o3d.visualization.draw_geometries(geometries)


def display_original_cloud_2d(point_cloud, original_coords):
    # Extract x and y coordinates from the point cloud
    x_coords = point_cloud[:, 0]
    y_coords = point_cloud[:, 1]

    # Create a scatter plot of the points
    plt.scatter(x_coords, y_coords, c='b', s=1)

    # Create a red dot for each tree location
    for coords in original_coords:
        plt.scatter(coords[0], coords[1], c='r', s=1)

    # Set plot title and labels
    plt.title("Original Point Cloud (2D)")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Show the plot
    plt.show()


def plot_tree_locations(tree_indices, canopy_height_model):
    # Create x and y coordinates for the tree locations
    x_coords = tree_indices[1]
    y_coords = tree_indices[0]

    # Plot the canopy height model
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(canopy_height_model, cmap='viridis')

    # rotate the point cloud

    # Mark the tree locations with a tiny red dot
    for x, y in zip(x_coords, y_coords):
        rect = Rectangle((x, y), 3, 3, color='r', fill=False)
        ax.add_patch(rect)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Tree Locations')

    # Show the plot
    plt.show()


def plot_tree_locations_3d(tree_indices, canopy_height_model):
    # Create x, y, and z coordinates for the tree locations
    x_coords = tree_indices[1]
    y_coords = tree_indices[0]
    z_coords = canopy_height_model[tree_indices]

    # Create x and y coordinates for the grid
    x_grid = np.arange(0, canopy_height_model.shape[1])
    y_grid = np.arange(0, canopy_height_model.shape[0])

    # Create a meshgrid from the coordinates
    xx, yy = np.meshgrid(x_grid, y_grid)

    # Plot the canopy height model
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, canopy_height_model, cmap='viridis')

    # Plot the tree locations
    ax.scatter(x_coords, y_coords, z_coords, c='red', marker='o')

    # plot the ground

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')
    ax.set_title('Tree Locations')

    # Show the plot
    plt.show()


def compute_original_coordinates(tree_indices, resolution, min_coords, chm):
    # Extract the x and y indices of the tree locations
    x_indices = tree_indices[1]
    y_indices = tree_indices[0]

    # Compute the original x and y coordinates
    x_coords = ((x_indices + min_coords[1]) * resolution) - min_coords[1]
    y_coords = ((y_indices + min_coords[0]) * resolution) - min_coords[0]

    # Set z to 500
    z_coords = chm[tree_indices]

    # Combine the x, y, and z coordinates into a single array
    original_coords = np.vstack((x_coords, y_coords, z_coords)).transpose()

    return original_coords


def detect_tubular_form(point_cloud, query_coords, radius_threshold):
    # plot the point cloud in 3d
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='b', marker='.', s=1)

    # create a list to save the locations of the trees that are tubular
    tubular_tree_locations = []

    for query_coord in query_coords:
        # Filter points within a certain radius threshold from the query coordinate

        # Calculates the Euclidean distances between each point in the point cloud and the query coordinate using
        distances = np.linalg.norm(point_cloud - query_coord, axis=1)

        # Find the indices of the points that are within the radius threshold
        filtered_indices = np.where(distances <= radius_threshold)[0]
        # Extract the points that are within the radius threshold
        filtered_points = point_cloud[filtered_indices]

        if len(filtered_points) < 3:
            print("No points found within the radius threshold for the query coordinate:", query_coord)
            continue
        else:
           # print("Found", len(filtered_points), "points within the radius threshold for the query coordinate:")

            # Fit a line to the filtered points
            ransac = RANSACRegressor()
            ransac.fit(filtered_points[:, 2].reshape(-1, 1), filtered_points[:, 2])

            # Extract the inlier points
            inlier_mask = ransac.inlier_mask_
            inlier_points = filtered_points[inlier_mask]

            # Evaluate the fit and make a decision if it corresponds to a tubular form
            if len(inlier_points) > 0:
                x = inlier_points[:, 0]
                y = inlier_points[:, 1]
                z = inlier_points[:, 2]

                ransac.fit(np.column_stack((x, y)), z)

                # Calculate the residuals of the inlier points
                residuals = np.abs(ransac.predict(np.column_stack((x, y))) - z)
                max_residual = np.max(residuals)

                # Determine the threshold for classifying as tubular form
                threshold_tubular = 1.4  # Adjust this threshold based on your data

                if max_residual < threshold_tubular:
                    print("Tubular form detected for the query coordinate:", query_coord)
                    # save the query coordinate
                    tubular_tree_locations.append(query_coord)

            else:
                print("No inlier points found for the query coordinate:", query_coord)


    return tubular_tree_locations


if __name__ == '__main__':
    file_path = r"D:\TFG\Data\tiles\luxemburgo\luxemburgo\samples\test_tiles\part_3.las"
    # Read the LAS/LAZ file
    las = laspy.read(file_path)

    point_cloud = np.vstack((las.x, las.y, las.z)).transpose()

    # normalize the point cloud
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(point_cloud)

    # Get the total number of points
    total_points = las.header.point_count

    points_scaled2 = points_scaled

    resolution = 0.005  # meters

    # perform chm algorithm
    chm = compute_canopy_height_model(points_scaled2, resolution)

    threshold = 1  # Adjust this value to control tree detection sensitivity
    filter_sizer = 60  # Adjust this value to control tree detection sensitivity

    tree_indices = detect_trees(chm, threshold, filter_sizer)

    # plot_point_cloud_2d(point_cloud

    plot_tree_locations(tree_indices, chm)
    plot_tree_locations_3d(tree_indices, chm)

    min_coords = np.min(points_scaled, axis=0)
    max_coords = np.max(points_scaled, axis=0)

    tree_indices_2 = tree_indices

    original_coords = compute_original_coordinates(tree_indices_2, resolution, max_coords, chm)

    # unnormalize the point cloud and the original coordinates
    points_scaled_2 = scaler.inverse_transform(points_scaled)
    original_coords_2 = scaler.inverse_transform(original_coords)

    detection = detect_tubular_form(point_cloud, original_coords_2, 1.8)

    # print the number of trees detected and the number of trees that are tubular
    print("Number of trees detected:", len(tree_indices_2[0]))
    print("Number of trees that are tubular:", len(detection))

    # plot the point cloud in 3d
    display_original_cloud_with_dot(point_cloud, detection)
