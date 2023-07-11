import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import open3d as o3d


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


def plot_simulated_trees_o3d(tree_indices, cloud_points):
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_points)
    geometries = [pcd]

    # Create a cylinder for each tree location
    for tree_index in tree_indices:
        cylinder_height = np.linalg.norm(tree_index[2])
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.4, height=cylinder_height)
        # color the cylinder brown
        cylinder.paint_uniform_color([0.5, 0.25, 0])
        # Translate the cylinder to the tree location respective to the origin 0, 0, 0
        cylinder.translate(tree_index)
        cylinder.translate([0, 0, -cylinder_height / 2])
        # set the z value of the cylinder to 0
        geometries.append(cylinder)

    # create a cone for each tree starting at the tree location minus 10 meters
    for tree_index in tree_indices:
        cone = o3d.geometry.TriangleMesh.create_cone(radius=4, height=15)
        # color the cone green
        cone.paint_uniform_color([0, 0.5, 0])

        # Translate the cone to the tree location respective to the origin 0, 0, 0
        cone.translate(tree_index)

        geometries.append(cone)

    # Create an Open3D visualization window and add geometries
    o3d.visualization.draw_geometries(geometries)


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


def display_cloud(point_cloud):
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])


def display_original_cloud_with_centroids(point_cloud, centroids_detected, centroids_not_detected):
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    geometries = [pcd]

    # Create a red dot for each tree location
    for coords in centroids_detected:
        dot = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
        # color the dot green
        dot.paint_uniform_color([0, 1, 0])

        # Translate the dot to the tree location respective to the origin 0, 0, 0
        dot.translate(coords)

        geometries.append(dot)

    for c in centroids_not_detected:
        dot = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
        # color the dot red
        dot.paint_uniform_color([1, 0, 0])
        # Translate the dot to the tree location respective to the origin 0, 0, 0
        dot.translate(c)
        geometries.append(dot)

    # Create an Open3D visualization window and add geometries
    o3d.visualization.draw_geometries(geometries)


def display_slice_with_centroids(point_cloud, centroids_detected, detected):
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    geometries = [pcd]

    # Create a red dot for each tree location
    for coords in centroids_detected:
        dot = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
        # color the dot green
        if detected:
            dot.paint_uniform_color([0, 1, 0])
        else:
            dot.paint_uniform_color([1, 0, 0])

        # Translate the dot to the tree location respective to the origin 0, 0, 0
        dot.translate(coords)

        geometries.append(dot)
    # Create an Open3D visualization window and add geometries
    o3d.visualization.draw_geometries(geometries)


def display_centroids_in_2d(centroids):
    # Extract x and y coordinates from the point cloud
    x_coords = centroids[:, 0]
    y_coords = centroids[:, 2]

    # Create a scatter plot of the points
    plt.scatter(x_coords, y_coords, c='b', s=1)

    # Set plot title and labels
    plt.title("Centroids (2D)")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Show the plot
    plt.show()

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
