import csv

import laspy
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score
from anytree import Node, RenderTree
import open3d as o3d
from skspatial.objects import Line, Points

from detect_utils import check_linearity, gaussian_chm, compute_original_coordinates, calculate_r_squared
from visualization_utils import display_original_cloud_with_centroids, plot_tree_locations, \
    display_slice_with_centroids, display_centroids_in_2d, display_canopy_height_model, display_original_cloud_2d, \
    display_cloud

linearity_threshold = 0.02
min_tree_samples = 40
tree_detection_threshold = 0.2
filter_sizer = 70  # Adjust this value to control tree detection sensitivity
heightMap_resolution = 0.005  # meters
slice_height = 2  # meters
r_squared_threshold = 0.02


def compute_canopy_height_model(point_cloud_height_model, resolution):
    min_coords_height_model = np.min(point_cloud_height_model, axis=0)
    max_coords_height_model = np.max(point_cloud_height_model, axis=0)

    # Create a regular grid based on the resolution
    x_grid = np.arange(min_coords_height_model[0], max_coords_height_model[0], resolution)
    y_grid = np.arange(min_coords_height_model[1], max_coords_height_model[1], resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)

    # Flatten the grid coordinates
    xy_flat = np.column_stack((xx.ravel(), yy.ravel()))

    # Build a KD-tree from the point cloud
    kdtree = cKDTree(point_cloud_height_model[:, :2])

    # Query the nearest neighbors for each grid point
    _, indices = kdtree.query(xy_flat, k=1)

    # Compute the canopy height for each grid point relative to the ground
    canopy_height = point_cloud_height_model[indices, 2]

    # Reshape the canopy height to match the grid shape
    canopy_height_model = canopy_height.reshape(xx.shape)
    return canopy_height_model


def detect_trees(canopy_height_model, threshold, filter_size):
    # Apply local maximum filtering to find peaks
    neighborhood_size = (filter_size, filter_size)
    local_max = maximum_filter(canopy_height_model, footprint=np.ones(neighborhood_size), mode='constant')
    tree_mask = (canopy_height_model == local_max) & (canopy_height_model > threshold)

    # Find the indices of the tree locations
    tree_indexes = np.where(tree_mask)

    return tree_indexes


def slice_and_get_centroids(points, slice_height):
    # Sort points by height in ascending order
    sorted_points = points[points[:, 2].argsort()]

    # Define slice height and number of slices
    min_height = sorted_points[0, 2]
    max_height = sorted_points[-1, 2]
    num_slices = int((max_height - min_height) / slice_height) + 1

    centroids = []  # Store centroids for each slice

    for i in range(num_slices):
        # Define lower and upper height boundaries for the current slice
        lower_height = min_height + i * slice_height
        upper_height = lower_height + slice_height

        # Extract points within the height range of the current slice
        slice_points = sorted_points[(sorted_points[:, 2] >= lower_height) &
                                     (sorted_points[:, 2] < upper_height)]

        # Calculate the central point of the slice if there are points in the slice
        if slice_points.shape[0] > 0:
            centroid = np.mean(slice_points, axis=0)
            # Append centroid to the list
            centroids.append(centroid)

    return np.array(centroids)


def slice_and_get_blobs(points, slice_height):
    # Sort points by height in ascending order
    global r_squared_slice, centroids_slice
    sorted_points = points[points[:, 2].argsort()]

    # Define slice height and number of slices
    min_height = sorted_points[0, 2]
    max_height = sorted_points[-1, 2]
    num_slices = int((max_height - min_height) / slice_height) + 1

    centroids = []  # Store centroids for each slice

    for i in range(num_slices - 6):
        # Define lower and upper height boundaries for the current slice
        lower_height = min_height + i * slice_height
        upper_height = lower_height + slice_height

        # Extract points within the height range of the current slice
        slice_points = sorted_points[(sorted_points[:, 2] >= lower_height) &
                                     (sorted_points[:, 2] < upper_height)]

        # cluster the points in the slice
        if slice_points.shape[0] > 0:
            # cluster the points
            cluster = DBSCAN(eps=0.7).fit(slice_points[:, :2])
            # get the labels
            labels = cluster.labels_
            # get the number of clusters
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            # get the unique labels
            unique_labels = set(labels)

            r_squared_slice = []
            centroids_slice = []

            # if there is more than one cluster and the number of centroids is greater than 1
            if n_clusters > 1 and len(centroids) > 1:
                # get the centroid of each cluster
                for label in unique_labels:
                    if label != -1:
                        # get the points in the cluster
                        cluster_points = slice_points[labels == label]
                        # compute the centroid
                        centroid = np.mean(cluster_points, axis=0)
                        centroids_slice.append(centroid)

                        # create a new list of centroids with the current centroid and the previous centroids
                        other_centroids = np.concatenate((centroids, centroids_slice), axis=0)

                        # degree = 1

                        # create polynomial features
                        # poly_features = PolynomialFeatures(degree=degree)
                        # x_poly = poly_features.fit_transform(other_centroids[:, 0].reshape(-1, 1))

                        # regression_model = LinearRegression()
                        # regression_model.fit(x_poly, other_centroids[:, 1].reshape(-1, 1))

                        # y_pred = regression_model.predict(x_poly)

                        # r_squared_slice = r2_score(other_centroids[:, 1].reshape(-1, 1), y_pred)
                        r_squared_slice = check_linearity(np.array(centroids)[:, :2])
                        best_cluster_idx = np.argmax(r_squared_slice)
                        best_centroid = centroids_slice[best_cluster_idx]
                        centroids.append(best_centroid)
            else:
                centroids.append(np.mean(slice_points, axis=0))

    return np.array(centroids)


def slice_and_get_points_tree(points, slice_height):
    level = 0  # Variable to keep track of the current level of the tree

    sorted_points = points[points[:, 2].argsort()]  # Sort points by height in ascending order

    # Define slice height and number of slices
    min_height = sorted_points[0, 2]
    max_height = sorted_points[-1, 2]
    num_slices = int((max_height - min_height) / slice_height) + 1

    root = Node(str(level), score=0, centroid=[])  # Create root node for the tree
    level += 1

    prev_nodes = [root]  # Store nodes from the previous level

    for i in range(num_slices - 4):

        subLevel = 0  # Reset sub-level to 0 for each slice

        # Define lower and upper height boundaries for the current slice
        lower_height = min_height + i * slice_height
        upper_height = lower_height + slice_height

        # Extract points within the height range of the current slice
        slice_points = sorted_points[(sorted_points[:, 2] >= lower_height) &
                                     (sorted_points[:, 2] < upper_height)]

        # temporal list of nodes
        temp_nodes = []

        if len(slice_points) > 0:

            # cluster the points
            cluster = DBSCAN(eps=0.3).fit(slice_points[:, :2])
            # get the labels
            labels = cluster.labels_
            # get the number of clusters
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            # get the unique labels
            unique_labels = set(labels)

            centroid = None
            tree_centroids = []

            if n_clusters > 0:
                for label in unique_labels:
                    if label != -1:
                        # get the points in the cluster
                        cluster_points = slice_points[labels == label]
                        # compute the centroid
                        centroid = np.mean(cluster_points, axis=0)
                        tree_centroids.append(centroid)

                        for node in prev_nodes:
                            # get the parent node of the current until the root
                            parent_node = node
                            while parent_node.parent is not None:
                                # get the centroid of the parent node
                                if parent_node.parent.name != "0":
                                    tree_centroids.append(parent_node.centroid)
                                parent_node = parent_node.parent

                            score = calculate_r_squared(np.array(tree_centroids))
                            node = Node(str(level) + "-" + str(subLevel), parent=node, score=score, centroid=centroid)
                            temp_nodes.append(node)
                    subLevel += 1
            else:
                centroid = np.mean(slice_points, axis=0)
                tree_centroids.append(centroid)
                for node in prev_nodes:
                    # get the parent node of the current until the root
                    parent_node = node
                    while parent_node.parent is not None:
                        # get the centroid of the parent node
                        if parent_node.parent.name != "0":
                            tree_centroids.append(parent_node.centroid)
                        parent_node = parent_node.parent

                    score = calculate_r_squared(np.array(tree_centroids))
                    node = Node(str(level) + "-" + str(subLevel), parent=node, score=score, centroid=centroid)
                    temp_nodes.append(node)

            prev_nodes = temp_nodes
            level += 1

    # get a list of all the centroids
    all_centroids = []

    for pre, fill, node in RenderTree(root):
        print("%s%s  %s" % (pre, node.name, node.score))
        if node.name != "0":
            all_centroids.append(node.centroid)
    print(all_centroids)

    file = r"C:\Users\Xiao\PycharmProjects\pythonProject\csv\centroids.csv"

    pc = []

    with open(file, 'a', newline='') as file:
        writer = csv.writer(file)
        for c in all_centroids:
            writer.writerow(c)
            pc.append(c)
        writer.writerow("\n")
    # display_slice_with_centroids(points, all_centroids)

    best_score = 0
    best_node = None
    best_centroids = []

    for node in prev_nodes:
        if node.score > best_score:
            best_score = node.score
            best_node = node
    # print("Best score: ", best_score)
    # print("Best node: ", best_node.name)

    while best_node.parent is not None:
        if len(best_node.parent.centroid) > 0:
            best_centroids.append(best_node.parent.centroid)
            best_node = best_node.parent
        else:
            break

    return best_centroids


def detect_tubular_form2(point_cloud, query_coords, radius_threshold,num_slices):
    # create a list to save the locations of the trees that are tubular
    tubular_tree_locations = []
    no_tubular_tree_locations = []

    for query_coord in query_coords:

        # Calculate the distances in 2D space (only considering x and y coordinates)
        distances = np.linalg.norm(point_cloud[:, :2] - query_coord[:2], axis=1)

        # Find the indices of the points that are within the radius threshold
        filtered_indices = np.where(distances <= radius_threshold)[0]

        filtered_points = point_cloud[filtered_indices]
        if len(filtered_points) < min_tree_samples:
            # print("No points found within the radius threshold for the query coordinate:", query_coord)
            continue
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(filtered_points)


            centroids = slice_and_get_points_tree(filtered_points, slice_height)

            centroids = np.array(centroids)

            # check if centroids contains more than 3 points and not NaN
            if len(centroids) > 4 and not np.isnan(centroids).any():

                r2 = calculate_r_squared(centroids)

                is_line = r2 > r_squared_threshold

                if is_line:
                    print("detected tree at: ", query_coord, " with r2: ", r2)
                    #o3d.visualization.draw_geometries([pcd])
                    #display_slice_with_centroids(filtered_points, centroids, True)
                    tubular_tree_locations.append(query_coord)
                else:
                    print("not detected tree at: ", query_coord, " with r2: ", r2)
                    #display_slice_with_centroids(filtered_points, centroids, False)
                    no_tubular_tree_locations.append(query_coord)

    # print(tubular_tree_locations)
    # print(no_tubular_tree_locations)
    display_original_cloud_with_centroids(point_cloud, tubular_tree_locations, no_tubular_tree_locations)

    return tubular_tree_locations


def detect_tubular_form(point_cloud, query_coords, radius_threshold):
    # create a list to save the locations of the trees that are tubular
    tubular_tree_locations = []

    for query_coord in query_coords:
        # Filter points within a certain radius threshold from the query coordinate

        # Calculate the distances in 2D space (only considering x and y coordinates)
        distances = np.linalg.norm(point_cloud[:, :2] - query_coord[:2], axis=1)

        # Find the indices of the points that are within the radius threshold
        filtered_indices = np.where(distances <= radius_threshold)[0]

        # Extract the points that are within the radius threshold
        filtered_points = point_cloud[filtered_indices]

        if len(filtered_points) < 3:
            print("No points found within the radius threshold for the query coordinate:", query_coord)
            continue
        else:

            ransac = RANSACRegressor()
            ransac.fit(filtered_points[:, 2].reshape(-1, 1), filtered_points[:, 2])

            # Extract the inlier points
            inlier_mask = ransac.inlier_mask_
            inlier_points = filtered_points[inlier_mask]

            centroids = slice_and_get_blobs(filtered_points, 8)
            print("Found", len(centroids), "centroids for the query coordinate:")
            print(centroids)

            # Create a 3D scatter plot of the centroids

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
    # Open cloud of points with ground
    # file_path = r"D:\TFG\Data\tiles\luxemburgo\luxemburgo\samples\test_tiles\part_7.las"
    file_path = r"C:\Users\Xiao\PycharmProjects\pythonProject/part_1.las"

    # Read the LAS/LAZ file
    las = laspy.read(file_path)
    point_cloud = np.vstack((las.x, las.y, las.z)).transpose()

    # Open cloud of points without ground
    # file_path_no_ground = r"D:\TFG\Data\tiles\luxemburgo\luxemburgo\samples\test_tiles\hag_no_ground.las"
    file_path_no_ground = r"C:\Users\Xiao\PycharmProjects\pythonProject/part_1_no_ground.las"
    # Read the LAS/LAZ file
    las_no_ground = laspy.read(file_path_no_ground)
    point_cloud_no_ground = np.vstack((las_no_ground.x, las_no_ground.y, las_no_ground.z)).transpose()

    # Open cloud of points only ground
    file_path_ground = r"C:\Users\Xiao\PycharmProjects\pythonProject/part_1_no_ground.las"
    # Read the LAS/LAZ file
    las_ground = laspy.read(file_path_ground)

    point_cloud_ground = np.vstack((las_ground.x, las_ground.y, las_ground.z)).transpose()

    # normalize the point cloud
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(point_cloud)

    # Get the total number of points
    total_points = las.header.point_count

    points_scaled2 = points_scaled

    # perform chm algorithm
    chm_or = compute_canopy_height_model(points_scaled2, heightMap_resolution)

    display_canopy_height_model(chm_or, heightMap_resolution)
    display_original_cloud_2d(chm_or, points_scaled2)

    chm = gaussian_chm(chm_or, 2)

    tree_indices = detect_trees(chm, tree_detection_threshold, filter_sizer)

    print("Found", len(tree_indices[0]), "tree locations in first Step.")

    #plot_point_cloud_2d(point_cloud
    plot_tree_locations(tree_indices, chm_or)
    plot_tree_locations(tree_indices, chm_or)
    #plot_tree_locations(tree_indices, chm)
    display_cloud(point_cloud)
    min_coords = np.min(points_scaled, axis=0)
    max_coords = np.max(points_scaled, axis=0)

    tree_indices_2 = tree_indices

    original_coords = compute_original_coordinates(tree_indices_2, heightMap_resolution, max_coords, chm)

    # denormalize the point cloud and the original coordinates
    points_scaled_2 = scaler.inverse_transform(points_scaled)
    original_coords_2 = scaler.inverse_transform(original_coords)
    # display_original_cloud_with_dot(point_cloud_no_ground, original_coords_2)

    detection = detect_tubular_form2(point_cloud_no_ground, original_coords_2, 2)
    print("Found", len(detection), "tree locations in second Step.")

    # print the number of trees detected and the number of trees that are tubular
    print("Number of trees detected:", len(detection[0]))
