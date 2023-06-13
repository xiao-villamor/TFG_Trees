## create a main function
import laspy
import pylas
import numpy as np



def split_cloud_into_squares(cloud_points, num_tiles):
    # Calculate the bounding box of the cloud
    min_coords = np.min(cloud_points, axis=0)
    max_coords = np.max(cloud_points, axis=0)
    bbox_size = max_coords - min_coords

    # Calculate the area of each tile
    total_area = bbox_size[0] * bbox_size[1]
    tile_area = total_area / num_tiles

    # Calculate the width and height of each tile
    tile_size = np.sqrt(tile_area)
    num_tiles_x = int(np.ceil(bbox_size[0] / tile_size))
    num_tiles_y = int(np.ceil(bbox_size[1] / tile_size))

    # Calculate the actual width and height of each tile
    actual_width = bbox_size[0] / num_tiles_x
    actual_height = bbox_size[1] / num_tiles_y

    # Split the cloud into tiles
    split_tiles = []
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            # Calculate the coordinates for the current tile
            start_x = i * actual_width + min_coords[0]
            end_x = start_x + actual_width
            start_y = j * actual_height + min_coords[1]
            end_y = start_y + actual_height

            # Extract the points within the current tile
            mask = (cloud_points[:, 0] >= start_x) & (cloud_points[:, 0] < end_x) & \
                   (cloud_points[:, 1] >= start_y) & (cloud_points[:, 1] < end_y)
            tile_points = cloud_points[mask]

            # Add the current tile to the list
            split_tiles.append(tile_points)


    # Save each part as a separate LAS file
    for i, part in enumerate(split_tiles):
        outfile = f"part_{i + 1}.las"

        # Extract coordinates from each part
        x = part[:, 0]
        y = part[:, 1]
        z = part[:, 2]  # assuming 3D points

        # Create a new LAS file
        out_las = laspy.create(file_version="1.2", point_format=2)

        # Set the point coordinates
        out_las.x = x
        out_las.y = y
        out_las.z = z

        out_las.write(outfile)

        print(f"Part {i + 1} saved as {outfile}.")


if __name__ == '__main__':
    file_path = r"D:\TFG\Data\tiles\luxemburgo\luxemburgo\samples\test_tiles\part_1.las"
    las_file = pylas.read(file_path)
    las_file = pylas.convert(las_file)

    # get points
    points = np.column_stack([las_file.x, las_file.y, las_file.z])

    split_cloud_into_squares(points, 4)
