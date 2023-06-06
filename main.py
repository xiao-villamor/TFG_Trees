import numpy as np
import laspy
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree, NearestNeighbors

from utils import display_point_cloud, load_las


def remove_ground_ml_laz(laz_file, test_size=0.2, random_state=42):
    # Read the LAS/LAZ file using laspy
    las = laspy.read(laz_file)

    # Extract point cloud data
    point_cloud = np.vstack((las.x, las.y, las.z)).T

    # Extract labels (ground or non-ground) from laspy classifications
    labels = las.classification

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(point_cloud, labels, test_size=test_size,
                                                      random_state=random_state)

    # Initialize a random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions on the validation set
    val_predictions = clf.predict(X_val)

    # Evaluate the model
    accuracy = accuracy_score(y_val, val_predictions)
    print("Validation Accuracy:", accuracy)

    # Remove ground points based on the trained model
    ground_indices = np.where(clf.predict(point_cloud) == 2)[0]
    non_ground_points = np.delete(point_cloud, ground_indices, axis=0)

    return non_ground_points


def remove_ground(points, k_neighbors=5, distance_threshold=0.2):
    # Build KDTree for fast nearest neighbor search
    tree = KDTree(points[:, :2])

    # Find indices of ground points
    ground_indices = []
    for i, point in enumerate(points):
        # Query k-nearest neighbors
        _, indices = tree.query([point[:2]], k=k_neighbors)

        # Calculate mean vertical distance to neighbors
        mean_z = np.mean(points[indices, 2])
        vertical_distance = np.abs(point[2] - mean_z)

        # Check if the point is likely to be ground
        if vertical_distance < distance_threshold:
            ground_indices.append(i)

    # Filter out ground points
    non_ground_points = np.delete(points, ground_indices, axis=0)

    return non_ground_points


def detect_trees(point_cloud, threshold, ransac_iter, min_points):
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    # Set the specified parameters
    search_radius = 6.0
    num_sample_points = 150
    num_iterations = 50
    inlier_threshold = 0.15
    acceptable_model_size = 15
    max_elevation_diff = 1.0

    # Perform nearest neighbor search
    neighbors = NearestNeighbors(n_neighbors=num_sample_points).fit(np.column_stack((x, y, z)))
    distances, indices = neighbors.radius_neighbors(np.column_stack((x, y, z)), radius=search_radius)

    # Perform RANSAC plane fitting to detect trees
    tree_points = []
    for i in range(len(point_cloud)):
        if len(indices[i]) < num_sample_points:
            continue

        sample_points = point_cloud[indices[i]]
        x_sample = sample_points[:, 0]
        y_sample = sample_points[:, 1]
        z_sample = sample_points[:, 2]

        # Estimate the plane normal using linear regression
        model = linear_model.RANSACRegressor(
            base_estimator=None,
            residual_threshold=inlier_threshold,
            max_trials=num_iterations,
            min_samples=acceptable_model_size
        )
        model.fit(np.column_stack((x_sample, y_sample)), z_sample)

        # Compute the residuals for each point
        residuals = np.abs(model.predict(np.column_stack((x, y))) - z)

        # Filter out points based on residuals and other criteria
        mask = (
                residuals <= inlier_threshold and
                np.abs(z - np.mean(z_sample)) <= max_elevation_diff
        )

        if np.sum(mask) >= acceptable_model_size:
            tree_points.append(point_cloud[i])

    tree_points = np.array(tree_points)

    return tree_points


if __name__ == '__main__':
    file_path = r"D:\TFG\Data\tiles\luxemburgo\luxemburgo\samples\test_tiles\test_tile.las"
    save_path = r"D:\TFG\Data\tiles\luxemburgo\luxemburgo\samples\python_test.las"

    points = load_las(file_path)

    trees = detect_trees(points, threshold=0.2, ransac_iter=100, min_points=100)
    # print a text summary of the file
    display_point_cloud(trees)
