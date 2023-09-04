import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor

from detect_utils import check_linearity


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

                        other_centroids = np.concatenate((centroids, centroids_slice), axis=0)

                        r_squared_slice = check_linearity(np.array(centroids)[:, :2])
                        best_cluster_idx = np.argmax(r_squared_slice)
                        best_centroid = centroids_slice[best_cluster_idx]
                        centroids.append(best_centroid)
            else:
                centroids.append(np.mean(slice_points, axis=0))

    return np.array(centroids)

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
