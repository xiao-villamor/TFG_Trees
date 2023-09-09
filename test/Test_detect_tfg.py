import csv
import os

import numpy as np
from sklearn.preprocessing import StandardScaler

from visualization_utils import plot_tree_locations, display_slice_with_centroids, \
    display_original_cloud_with_centroids, display_canopy_height_model
from detect_utils import check_accuracy
from detect import compute_canopy_height_model, gaussian_chm, detect_trees_in_max, compute_original_coordinates, \
    detect_tubular_form2

from utils import load_las, measure_execution_time


@measure_execution_time
def test_detection():

    files = os.listdir('D:\Data\Datatest\Data\ground')
    #files = os.listdir('D:\Data\Datatest\Data 1\LIDAR2019_NdP_57000_105000_EPSG2169\ground')
    #files = os.listdir('D:\Data\Datatest\Data 2\LIDAR2019_NdP_81500_82500_EPSG2169\ground')

    num_files = len(files)
    total_accuracy = 0.0
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    f1_score = 0.0
    precision = 0.0
    recall = 0.0
    specificity = 0.0
    treeNum = 0

    for file in files:
        point_cloud = load_las("D:\Data\Datatest\Data\ground\\" + file)
        #point_cloud = load_las("D:\Data\Datatest\Data 1\LIDAR2019_NdP_57000_105000_EPSG2169\ground\\" + file)
        #point_cloud = load_las("D:\Data\Datatest\Data 2\LIDAR2019_NdP_81500_82500_EPSG2169\ground\\" + file)

        # open csv file
        scaler = StandardScaler()
        points_scaled = scaler.fit_transform(point_cloud)
        points_scaled2 = points_scaled

        resolution = 0.005  # meters

        # perform chm algorithm
        chm = compute_canopy_height_model(points_scaled2, resolution)
        chm = gaussian_chm(chm, 1.7)

        threshold = 0.3  # Adjust this value to control tree detection sensitivity
        filter_sizer = 80  # Adjust this value to control tree detection sensitivity

        tree_indices = detect_trees_in_max(chm, threshold, filter_sizer)

        max_coords = np.max(points_scaled, axis=0)

        tree_indices_2 = tree_indices

        original_coords = compute_original_coordinates(tree_indices_2, resolution, max_coords, chm)

        original_coords_2 = scaler.inverse_transform(original_coords)

        # plot_tree_locations([0,0,0],chm)
        #plot_tree_locations(tree_indices, chm_or)
        #plot_tree_locations(tree_indices, chm)
        # Open cloud of points without ground

        file_path_no_ground = r"D:\Data\Datatest\Data\no_ground\\" + file
        #file_path_no_ground = r"D:\Data\Datatest\Data 1\LIDAR2019_NdP_57000_105000_EPSG2169\no_ground\\" + file
        #file_path_no_ground = r"D:\Data\Datatest\Data 2\LIDAR2019_NdP_81500_82500_EPSG2169\no_ground\\" + file
        # Read the LAS/LAZ file

        point_cloud_no_ground = load_las(file_path_no_ground)

        detection2, no_tree_indices2 = detect_tubular_form2(point_cloud_no_ground, original_coords_2, 2)

        # remove the extension .las
        fileN = file + '.csv'

        # function check_accuracy returns accuracy, TP, FP, FN, num_trees, f1_score accumulate all the values in her respective variable

        total_accuracyT, TPT, FPT, FNT, TNT, treeNumT, f1_scoreT, precisionT, recallT, specificityT = check_accuracy(detection2,no_tree_indices2 ,fileN)
        total_accuracy += total_accuracyT
        TP += TPT
        FP += FPT
        FN += FNT
        TN += TNT
        treeNum += treeNumT
        f1_score += f1_scoreT
        precision += precisionT
        recall += recallT
        specificity += specificityT




    # print("Total accuracy: ", total_accuracy / num_files)
    #directory = r'D:\Data\Datatest\Data 1\LIDAR2019_NdP_57000_105000_EPSG2169\Results'
    #directory = r'D:\Data\Datatest\Data 2\LIDAR2019_NdP_81500_82500_EPSG2169\Results'
    directory = r'D:\Data\Datatest\Data\Results'
    file_path = os.path.join(directory, 'total_metrics.csv')
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
        writer.writerow([treeNum])
        writer.writerow(["detected Trees"])
        writer.writerow([TP + FP])
        writer.writerow(["not detected Trees"])
        writer.writerow([FN + TN])
        writer.writerow(["f1_score"])
        writer.writerow([f1_score / num_files])
        writer.writerow(["Accuracy"])
        writer.writerow([total_accuracy / num_files])
        writer.writerow(["precision"])
        writer.writerow([precision / num_files])
        writer.writerow(["recall"])
        writer.writerow([recall / num_files])
        writer.writerow(["specificity"])
        writer.writerow([specificity / num_files])