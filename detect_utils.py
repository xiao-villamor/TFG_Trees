import os
from datetime import datetime

import csv

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from skimage.restoration import denoise_bilateral
import tensorflow as tf


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


def calculate_r_squared(points):
    if len(points) < 3:
        return 0.0

        # Extract x, y, and z coordinates from the points
    x = np.array([point[0] for point in points])
    y = np.array([point[1] for point in points])
    z = np.array([point[2] for point in points])

    # Create design matrix
    X = np.column_stack((x, y, np.ones(len(points))))

    # Fit a linear regression plane
    coefficients, _, _, _ = np.linalg.lstsq(X, z, rcond=None)

    # Calculate the predicted z values on the plane
    predicted_z = np.dot(X, coefficients)

    # Calculate the sum of squared errors
    ss_total = np.sum((z - np.mean(z)) ** 2)
    ss_residual = np.sum((z - predicted_z) ** 2)

    # Calculate R-squared
    r_squared = 1.0 - (ss_residual / ss_total)

    return r_squared


def old_calculate_r_squared(points):
    if len(points) < 2:
        return 0.0

    # Extract x, y, and z coordinates from the points
    x = np.array([point[0] for point in points])
    y = np.array([point[1] for point in points])
    z = np.array([point[2] for point in points])

    # Fit a linear regression plane
    coefficients = np.polyfit(x, y, 1)
    slope_x, intercept_x = coefficients
    coefficients = np.polyfit(x, z, 1)
    slope_z, intercept_z = coefficients

    # Calculate the predicted y and z values on the plane
    predicted_y = slope_x * x + intercept_x
    predicted_z = slope_z * x + intercept_z

    # Calculate the sum of squared errors
    ss_total = np.sum((y - np.mean(y)) ** 2 + (z - np.mean(z)) ** 2)
    ss_residual = np.sum((y - predicted_y) ** 2 + (z - predicted_z) ** 2)

    # Calculate R-squared
    r_squared = 1.0 - (ss_residual / ss_total)

    return r_squared


def check_accuracy(points, point_no_trees, file):
    # load the csv with the expected tree locations
    fileD = "D:\Data\Datatest\Data\groundT\\" + file
    #fileD = "D:\Data\Datatest\Data 1\LIDAR2019_NdP_57000_105000_EPSG2169\groundT\\" + file
    #fileD = "D:\Data\Datatest\Data 2\LIDAR2019_NdP_81500_82500_EPSG2169\groundT\\" + file

    expected_tree_locations = np.loadtxt(fileD, delimiter=',')
    num_trees = expected_tree_locations.shape[0]

    treeFound = []
    notDetect = []


    for point in points:
        x = point[0]
        y = point[1]

        for expected_tree_location in expected_tree_locations:
            x_expected = expected_tree_location[0]
            y_expected = expected_tree_location[1]

            indices = np.where((expected_tree_locations == expected_tree_location))

            # check if the point is near to the expected tree location
            if abs(x - x_expected) < 15 and abs(y - y_expected) < 15:
                treeFound.append(point)
                # delete the expected tree location from expected_tree_locations
                expected_tree_locations = np.delete(expected_tree_locations, indices[0][0], 0)
                break



    # find the number of trees that are not detected as trees but are trees ussing point_no_trees
    for pointN in point_no_trees:
        x = pointN[0]
        y = pointN[1]


        for expected_tree_location in expected_tree_locations:
            x_expected = expected_tree_location[0]
            y_expected = expected_tree_location[1]

            indices = np.where((expected_tree_locations == expected_tree_location))

            # check if the point is near to the expected tree location
            if abs(x - x_expected) < 15 and abs(y - y_expected) < 15:
                notDetect.append(pointN)
                expected_tree_locations = np.delete(expected_tree_locations, indices[0][0], 0)
                break

    # Number of points that are trees and detected  (TP)
    TP = len(treeFound)

    # FP  number of points that are not trees but detected as trees
    FP = len(points) - TP

    # FN number of points that are trees but not detected as trees
    FN = len(notDetect)

    # TN number of points that are not trees and not detected as trees
    TN = len(point_no_trees) - FN


    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calculate accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    # get the actual timestamp
    now = datetime.now()

    directory = r'D:\Data\Datatest\Data\Results'
    #directory = r'D:\Data\Datatest\Data 1\LIDAR2019_NdP_57000_105000_EPSG2169\Results'
    #directory = r'D:\Data\Datatest\Data 2\LIDAR2019_NdP_81500_82500_EPSG2169\Results'
    file_path = os.path.join(directory, file + 'accuracy.csv')

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["True Positives"])
        writer.writerow([TP])
        writer.writerow(["False Positives"])
        writer.writerow([FP])
        writer.writerow(["False Negatives"])
        writer.writerow([FN])
        writer.writerow(["True Negatives"])
        writer.writerow([TN])
        writer.writerow(["Expected Trees"])
        writer.writerow([num_trees])
        writer.writerow(["Detected Trees"])
        writer.writerow([TP + FP])
        writer.writerow(["Detected Trees length"])
        writer.writerow([len(points)])
        writer.writerow(["Accuracy"])
        writer.writerow([accuracy])
        writer.writerow(["Precision"])
        writer.writerow([precision])
        writer.writerow(["Specificity"])
        writer.writerow([specificity])
        writer.writerow(["Recall"])
        writer.writerow([recall])
        writer.writerow(["f1_score"])
        writer.writerow([f1_score])
        writer.writerow(["Date"])
        writer.writerow([now.strftime("%d/%m/%Y %H:%M:%S")])

    # return all the values
    return accuracy, TP, FP, FN,TN, num_trees, f1_score, precision, recall, specificity


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
