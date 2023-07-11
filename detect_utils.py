import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from skimage.restoration import denoise_bilateral
import tensorflow as tf
from tensorflow.keras import layers

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


def check_linearity(points):

    # Calculate the slope between consecutive points
    slopes = []
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        slope = (y2 - y1) / (x2 - x1)
        slopes.append(slope)

    # Calculate the average difference between consecutive slopes
    avg_diff = sum(abs(slopes[i] - slopes[i + 1]) for i in range(len(slopes) - 1)) / (len(slopes) - 1)

    # Inverse the average difference to get linearity score
    linearity_score = 1.0 / (1.0 + avg_diff)

    return linearity_score


def check_accuracy(points):
    # load the csv with the expected tree locations
    expected_tree_locations = np.loadtxt(r'C:\Users\Xiao\PycharmProjects\pythonProject\groundT\part_1.las.csv', delimiter=',')
    pcl = []

    for point in points:
        x = point[0]
        y = point[1]
        # print('point')
        # print(x, y)

        for expected_tree_location in expected_tree_locations:
            x_expected = expected_tree_location[0]
            y_expected = expected_tree_location[1]
            # print(x_expected, y_expected)

            indices = np.where((expected_tree_locations == expected_tree_location))

            # check if the point is near to the expected tree location
            if abs(x - x_expected) < 25 and abs(y - y_expected) < 25:
                pcl.append(point)
                # delete the expected tree location from expected_tree_locations
                # print('found a tree')
                break
    # print(len(pcl))
    # get the number of points that are trees
    num_points_trees = len(pcl)

    # get the number of points that are not trees
    num_points_not_trees = len(points) - num_points_trees

    # get the number of points that are trees from the csv
    num_points_trees = expected_tree_locations.shape[0]

    # return the accuracy
    return num_points_trees / (num_points_trees + num_points_not_trees)


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


def build_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(2,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model
