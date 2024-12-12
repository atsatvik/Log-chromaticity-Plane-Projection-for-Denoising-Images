import cv2
import numpy as np
import os
import plotly.graph_objects as go
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import scipy.stats
from ttictoc import tic, toc
from scipy.stats import entropy
import math
import tifffile
from concurrent.futures import ProcessPoolExecutor
from scipy import stats
import OpenEXR
import Imath
import imageio
import random


class processLogImage:
    def generate3DHistogram(
        self,
        image,
        image2=None,
        color_image=None,
        save_name="pcl.html",
        title="pcl",
        vector=None,
        planes=None,
        center_point=None,
        plane_size=200,
    ):
        x_vals, y_vals, z_vals = image[:, 0], image[:, 1], image[:, 2]
        x_vals = x_vals.flatten()
        y_vals = y_vals.flatten()
        z_vals = z_vals.flatten()

        if color_image is None:
            color_image = image

        r_color, g_color, b_color = (
            color_image[:, 0],
            color_image[:, 1],
            color_image[:, 2],
        )

        r_color = r_color.flatten()
        g_color = g_color.flatten()
        b_color = b_color.flatten()

        colors = []
        for red, blue, green in zip(r_color, g_color, b_color):
            colors.append((red, blue, green))

        sizes = 1

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    mode="markers",
                    marker=dict(size=sizes, color=colors, opacity=1),
                )
            ]
        )

        if image2 is not None:
            x_vals, y_vals, z_vals = image2[:, 0], image2[:, 1], image2[:, 2]
            x_vals = x_vals.flatten()
            y_vals = y_vals.flatten()
            z_vals = z_vals.flatten()
            fig.add_trace(
                go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    mode="markers",
                    marker=dict(size=sizes, color=colors, opacity=1),
                )
            )

        if vector is not None:
            for vec in vector:
                vector_start = vec[0]
                vector_end = vec[1]
                fig.add_trace(
                    go.Scatter3d(
                        x=[vector_start[0], vector_end[0]],
                        y=[vector_start[1], vector_end[1]],
                        z=[vector_start[2], vector_end[2]],
                        mode="lines",
                        line=dict(color="red", width=5),
                        name="Vector",
                    )
                )

        if planes is not None:
            for pl in planes:
                a, b, c, d = pl

                if center_point is not None:
                    x0, y0, z0 = center_point
                else:
                    x0, y0, z0 = 0, 0, 0  # Default center point

                # Create a meshgrid centered around the point (x0, y0)
                half_range = plane_size / 2
                xx, yy = np.meshgrid(
                    np.linspace(x0 - half_range, x0 + half_range, 10),
                    np.linspace(y0 - half_range, y0 + half_range, 10),
                )

                zz = (-a * xx - b * yy - d) / c

                fig.add_trace(
                    go.Surface(
                        x=xx,
                        y=yy,
                        z=zz,
                        colorscale="Viridis",
                        opacity=0.5,
                        name="Plane",
                    )
                )

        fig.update_layout(
            scene=dict(
                xaxis_title="Red",
                yaxis_title="Green",
                zaxis_title="Blue",
            ),
            title=title,
        )

        fig.write_html(save_name)
        print(f"PCL saved at {save_name}")

    def plot2Dpts(self, points, threshold=0.05, show_plot=False):
        points_array = np.array(points)

        tree = KDTree(points_array)
        updated_points_array = np.zeros_like(
            points_array
        )  # Array to store updated points
        visited = set()  # To keep track of visited points

        for i, point in enumerate(points_array):
            if i in visited:
                continue

            # Find indices of points within the threshold distance
            indices = tree.query_ball_point(point, threshold)

            # Merge points by taking their mean
            close_points = points_array[indices]
            merged_point = close_points.mean(axis=0)

            # Set the same merged point for all close points in the updated array
            for idx in indices:
                updated_points_array[idx] = merged_point

            # Mark all close points as visited
            visited.update(indices)

        # # Normalize the original points for plotting
        # x, y = points_array[:, 0], points_array[:, 1]
        # x = (x - np.min(x)) / (np.max(x) - np.min(x))
        # y = (y - np.min(y)) / (np.max(y) - np.min(y))

        # # Normalize the updated points for plotting
        # updated_x, updated_y = updated_points_array[:, 0], updated_points_array[:, 1]
        # updated_x = (updated_x - np.min(updated_x)) / (
        #     np.max(updated_x) - np.min(updated_x)
        # )
        # updated_y = (updated_y - np.min(updated_y)) / (
        #     np.max(updated_y) - np.min(updated_y)
        # )

        # if show_plot:
        #     # Plot original and updated points side by side
        #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        #     # Plot the original points
        #     ax1.scatter(x, y, color="blue", marker="o")
        #     ax1.set_title("Original Points")
        #     ax1.set_xlabel("X-axis")
        #     ax1.set_ylabel("Y-axis")
        #     ax1.grid()

        #     # Plot the updated (merged) points
        #     ax2.scatter(updated_x, updated_y, color="green", marker="o")
        #     ax2.set_title("Updated Points (After Merging)")
        #     ax2.set_xlabel("X-axis")
        #     ax2.set_ylabel("Y-axis")
        #     ax2.grid()

        #     plt.show()
        return updated_points_array

    def update_far_pixels(self, points, threshold_distance=0.1):
        H, W, _ = points.shape
        updated_points = points.copy()

        # Pad the array to handle edge cases (reflect padding)
        padded_points = np.pad(points, ((1, 1), (1, 1), (0, 0)), mode="reflect")

        for i in range(1, H + 1):
            for j in range(1, W + 1):
                # Extract the 3x3 region around the pixel
                region = padded_points[i - 1 : i + 2, j - 1 : j + 2, :]
                middle_pixel = padded_points[i, j, :]

                # Calculate the Euclidean distance between the middle pixel and other pixels in the region
                distances = np.linalg.norm(region - middle_pixel, axis=2)

                # Check how many pixels are close (within threshold distance)
                close_pixels_mask = distances <= threshold_distance
                close_pixels_count = (
                    np.sum(close_pixels_mask) - 1
                )  # Exclude middle pixel itself

                # If more than half of the pixels are far, replace the middle pixel with the average of close pixels
                if close_pixels_count <= 4:  # Half of the 8 neighboring pixels
                    close_pixels = region[close_pixels_mask]
                    if len(close_pixels) > 0:
                        updated_points[i - 1, j - 1, :] = np.mean(close_pixels, axis=0)

        return updated_points

    def estimateTransformationMat(self, normal_vector):
        # print("Vector from Shadow to Lit:", normal_vector)
        # Normalize the vector to get the normal of the plane
        normal_vector = normal_vector / np.linalg.norm(normal_vector)

        # Find two orthogonal vectors on the plane (using Gram-Schmidt process)
        # We can choose an arbitrary vector that is not parallel to the normal
        arbitrary_vector = (
            np.array([1, 0, 0])
            if not np.allclose(normal_vector, [1, 0, 0])
            else np.array([0, 1, 0])
        )

        # Use the Gram-Schmidt process to find a vector orthogonal to the normal
        v1 = arbitrary_vector - np.dot(arbitrary_vector, normal_vector) * normal_vector
        v1 = v1 / np.linalg.norm(v1)  # Normalize v1

        # Find another orthogonal vector to both normal_vector and v1 using cross product
        v2 = np.cross(normal_vector, v1)
        v2 = v2 / np.linalg.norm(v2)  # Normalize v2

        # Create the transformation matrix (3D to 2D projection onto the plane)
        transformation_matrix = np.vstack([v1, v2])  # Stack v1 and v2 into a 2x3 matrix
        return transformation_matrix

    def transformPoints(self, image, tf_mat):
        image = image.reshape(-1, 3)
        projected_points = np.dot(image, tf_mat.T)
        return projected_points

    def estimateGrayscaleImagefromXY(self, projected_points, image_shape):
        x_coords = projected_points[:, 0]
        y_coords = projected_points[:, 1]

        xx = x_coords + y_coords
        xx = (xx - xx.min()) / (xx.max() - xx.min()) * 255
        xx = xx.reshape(image_shape[0], image_shape[1]).astype(np.uint8)

        x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min()) * 255
        y_norm = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min()) * 255

        grayscale_values = 0.5 * (x_norm + y_norm)
        grayscale_image = grayscale_values.astype(np.uint8)

        grayscale_image = grayscale_image.reshape(image_shape[0], image_shape[1])

        # cv2.imshow("xx", xx)
        # cv2.imshow("grayscale_image", grayscale_image.astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return grayscale_image

    def findExtremePoints(self, points):
        if points.shape[0] < points.shape[1]:
            raise Exception(
                f"shape of points should be of form num_points,dim, got {points.shape}"
            )
        max_dist = 0
        farthest_points = None
        num_points = len(points)

        for i in range(num_points):
            for j in range(i + 1, num_points):
                dist = np.linalg.norm(points[i] - points[j])
                if dist > max_dist:
                    max_dist = dist
                    farthest_points = (points[i], points[j])

        return farthest_points, max_dist

    def getValBasedOnPatch(self, xy_map, tree, idx, threshold, min_neighbors, ch):
        points_array = xy_map.reshape(-1, ch)
        row = idx // 640
        col = idx % 640
        curr_pt = xy_map[row][col]

        patch = self.getPatch(xy_map, row, col, kernel_size=(5, 5)).reshape(-1, ch)
        valid_patch_point = []
        for pt in patch:
            patch_pt_NN_indices = tree.query_ball_point(pt, threshold)
            # ###############################################
            # distance_pts = np.sqrt(
            #     np.sum((points_array[patch_pt_NN_indices] - curr_pt) ** 2, axis=1)
            # )
            # distance_pts = np.round(distance_pts, 2)
            # non_overlapping_indices = np.where(distance_pts > 1e-6)[0]
            # patch_pt_NN_indices = np.array(patch_pt_NN_indices)[non_overlapping_indices]
            # ################################################

            if len(patch_pt_NN_indices) > min_neighbors:
                valid_patch_point.append(pt)
        return (
            np.mean(np.array(valid_patch_point), axis=0)
            if valid_patch_point
            else [0, 0, 0]
        )

    def filterPointsbyNearestNeighbor(
        self, points, threshold=0.05, min_neighbors=5, ch=None
    ):
        ch = 2 if ch is None else ch

        points_array = np.array(points)
        xy_map = points_array.reshape((480, 640, ch))

        tree = KDTree(points_array)

        updated_points_array = np.copy(points_array)

        visited = np.zeros(len(points_array), dtype=bool)

        for i, point in enumerate(points_array):
            if visited[i]:
                continue
            visited[i] = True

            indices = tree.query_ball_point(point, threshold)

            if len(indices) <= min_neighbors:
                updated_points_array[i] = self.getValBasedOnPatch(
                    xy_map, tree, i, threshold, min_neighbors, ch
                )
                # updated_points_array[i] = 0
            else:
                updated_points_array[indices] = points_array[indices]
                visited[indices] = True

        return updated_points_array

    def NEWWWWfilterPointsbyNearestNeighbor(
        self, points, threshold=0.05, min_neighbors=5, ch=None
    ):
        ch = 2 if ch is None else ch

        points_array = np.array(points)
        tree = KDTree(points_array)

        updated_points_array = np.copy(points_array)

        # To store the number of neighbors for each point
        neighbor_counts = np.zeros(len(points_array), dtype=int)

        # Calculate the number of neighbors within the threshold for each point
        for i, point in enumerate(points_array):
            indices = tree.query_ball_point(point, threshold)
            neighbor_counts[i] = len(indices)

        # Find the point with the maximum neighbors
        max_neighbors_index = np.argmax(neighbor_counts)
        max_neighbors_point = points_array[max_neighbors_index]

        # Now, for each point, replace points that don't meet the min_neighbors criterion
        for i, point in enumerate(points_array):
            if neighbor_counts[i] <= min_neighbors:
                updated_points_array[i] = max_neighbors_point

        return updated_points_array

    def projectPoint(self, point, plane_normal, D):
        point = np.array(point)
        plane_normal = np.array(plane_normal)

        # Step 1: Compute the 3D projected point onto the plane (same as before)
        distance = (np.dot(plane_normal, point) + D) / np.dot(
            plane_normal, plane_normal
        )
        projected_point_3D = point - distance * plane_normal

        # Step 2: Find two vectors (basis) that lie on the plane (orthogonal to the normal)
        # We can use the cross product with any arbitrary vector that is not parallel to the normal.
        arbitrary_vector = (
            np.array([1, 0, 0])
            if not np.allclose(plane_normal, [1, 0, 0])
            else np.array([0, 1, 0])
        )
        basis1 = np.cross(plane_normal, arbitrary_vector)
        basis1 /= np.linalg.norm(basis1)  # Normalize the vector

        basis2 = np.cross(plane_normal, basis1)  # The second basis vector
        basis2 /= np.linalg.norm(basis2)  # Normalize the vector

        # Step 3: Compute the 2D coordinates by projecting the 3D point onto the basis vectors
        local_2D_x = np.dot(projected_point_3D, basis1)  # Project onto basis1
        local_2D_y = np.dot(projected_point_3D, basis2)  # Project onto basis2

        return np.array([local_2D_x, local_2D_y])

    def reprojectXYtoRGB(self, pts_2d, constant_intensity=255):
        rgb_values = []

        x_vals, y_vals = pts_2d[:, 0], pts_2d[:, 1]

        R = ((x_vals - np.min(x_vals)) / (np.max(x_vals) - np.min(x_vals))) * 255
        G = ((y_vals - np.min(y_vals)) / (np.max(y_vals) - np.min(y_vals))) * 255

        # B = (R + G) / 2 - constant_intensity
        B = np.ones_like(R) * constant_intensity

        rgb = np.stack([R, G, B], axis=1)
        return rgb

    def reprojectXYtoOriginalColor(
        self,
        pts_2d,
        pts_3d,
        mask_pts,
        point_color_dict,
        minimum_same_mapped=100,
        intensity=None,
    ):
        rgb_values = []
        x_vals, y_vals = pts_2d[:, 0], pts_2d[:, 1]

        min_x, min_y = np.min(x_vals), np.min(y_vals)
        max_x, max_y = np.max(x_vals), np.max(y_vals)

        # x_vals = ((x_vals - min_x) / (max_x - min_x)).astype(int) * 255
        # y_vals = ((y_vals - min_y) / (max_y - min_y)).astype(int) * 255

        for i, (x, y) in tqdm(
            enumerate(zip(x_vals, y_vals)), desc="Reprojecting plane points to RGB"
        ):
            if mask_pts is not None and not mask_pts[i]:
                rgb_values.append(pts_3d[i, :])
                continue

            x = int(((x - min_x) / (max_x - min_x)) * 255)
            y = int(((y - min_y) / (max_y - min_y)) * 255)

            pt = tuple((x, y))

            # rgb_arr = np.array(list(point_color_dict[pt][0][1]))
            # rgb_intensity = np.sum(rgb_arr, axis=1)
            # sorted_indices = np.argsort(rgb_intensity)

            # max_idx = sorted_indices[-1]
            # R, G, B = rgb_arr[max_idx, :]
            R, G, B = point_color_dict[pt][0][1]

            # R_orig, G_orig, B_orig = pts_3d[i, :]
            # freq = point_color_dict[pt][1]
            # if freq < minimum_same_mapped:
            #     R, G, B = R_orig, G_orig, B_orig

            rgb_values.append([R, G, B])

        rgb_values = np.array(rgb_values)

        if intensity is not None:
            intensity_rgb_values = np.mean(rgb_values, axis=1)

            scale_factor_rgb_values = np.divide(
                intensity, intensity_rgb_values, where=intensity_rgb_values != 0
            )
            rgb_values = rgb_values * scale_factor_rgb_values[:, np.newaxis]
            rgb_values = rgb_values.astype(np.uint8)
        return rgb_values

    def temp_hehe(
        self,
        pts_3d,
        color_pts_3d,
        point_color_dict,
        mask_pts=None,
        minimum_same_mapped=100,
        intensity=None,
    ):
        rgb_values = []
        x_vals, y_vals, z_vals = pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2]
        min_x, min_y, min_z = np.min(x_vals), np.min(y_vals), np.min(z_vals)
        max_x, max_y, max_z = np.max(x_vals), np.max(y_vals), np.max(z_vals)

        for i, (x, y, z) in tqdm(
            enumerate(zip(x_vals, y_vals, z_vals)),
            desc="Reprojecting plane points to RGB",
        ):
            if mask_pts is not None and not mask_pts[i]:
                rgb_values.append(color_pts_3d[i, :])
                continue

            x = int(((x - min_x) / (max_x - min_x)) * 255)
            y = int(((y - min_y) / (max_y - min_y)) * 255)
            z = int(((z - min_z) / (max_z - min_z)) * 255)

            pt = tuple((x, y, z))

            rgb_arr = np.array(list(point_color_dict[pt][0]))
            rgb_intensity = np.sum(rgb_arr, axis=1)
            sorted_indices = np.argsort(rgb_intensity)

            max_idx = sorted_indices[-1]
            R, G, B = rgb_arr[max_idx, :]

            R_orig, G_orig, B_orig = color_pts_3d[i, :]
            freq = point_color_dict[pt][1]
            if freq < minimum_same_mapped:
                R, G, B = R_orig, G_orig, B_orig

            rgb_values.append([R, G, B])

        rgb_values = np.array(rgb_values)

        if intensity is not None:
            intensity_rgb_values = np.mean(rgb_values, axis=1)

            scale_factor_rgb_values = np.divide(
                intensity, intensity_rgb_values, where=intensity_rgb_values != 0
            )
            rgb_values = rgb_values * scale_factor_rgb_values[:, np.newaxis]
            rgb_values = rgb_values.astype(np.uint8)
        return rgb_values

    def calculate_kernel_average(self, image, row, col, kernel_size):
        kernel_size = kernel_size[0]
        half_kernel = kernel_size // 2

        height, width, channels = image.shape

        # Check if the kernel is within the image bounds, pad if necessary
        row_start = max(0, row - half_kernel)
        row_end = min(height, row + half_kernel + 1)
        col_start = max(0, col - half_kernel)
        col_end = min(width, col + half_kernel + 1)

        # Extract the 3x3 patch from the image
        patch = image[row_start:row_end, col_start:col_end, :]

        # Calculate the  value across the 3x3 patch for each channel
        kernel_average = np.mean(patch, axis=(0, 1))

        return kernel_average

    def getPatch(self, image, row, col, kernel_size):
        kernel_size = kernel_size[0]
        half_kernel = kernel_size // 2

        height, width, channels = image.shape

        # Check if the kernel is within the image bounds, pad if necessary
        row_start = max(0, row - half_kernel)
        row_end = min(height, row + half_kernel + 1)
        col_start = max(0, col - half_kernel)
        col_end = min(width, col + half_kernel + 1)

        # Extract the 3x3 patch from the image
        patch = image[row_start:row_end, col_start:col_end, :]

        return patch

    def fitPlane(self, points, sufficient_inliers, max_iterations=1000, threshold=1e-2):
        best_plane = None
        best_inliers_count = 0
        best_inliers_mask = None

        num_points = points.shape[0]

        # RANSAC iterations
        for _ in tqdm(range(max_iterations), desc="Estimating best fit plane for RGB"):
            # Randomly select 3 points
            sample_indices = np.random.choice(num_points, 3, replace=False)
            sample_points = points[sample_indices]

            # Compute the normal of the plane from the 3 points
            v1 = sample_points[1] - sample_points[0]
            v2 = sample_points[2] - sample_points[0]
            normal = np.cross(v1, v2)

            # Ensure the normal is not degenerate (i.e., points are not collinear)
            if np.linalg.norm(normal) == 0:
                continue

            # Normalize the normal vector
            normal = normal / np.linalg.norm(normal)

            # Plane equation: ax + by + cz + d = 0
            # We can use any of the sample points to solve for d
            a, b, c = normal
            d = -np.dot(normal, sample_points[0])

            # Compute the distance from all points to the plane
            distances = np.abs(
                a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
            )

            # Find the inliers: points where the distance is less than the threshold
            inliers_mask = distances < threshold
            inliers_count = np.sum(inliers_mask)

            # Update the best model if the current one has more inliers
            if inliers_count > best_inliers_count:
                best_plane = (a, b, c, d)
                best_inliers_count = inliers_count
                best_inliers_mask = inliers_mask
                if best_inliers_count >= sufficient_inliers:
                    break

        return best_plane, best_inliers_mask

    def getNormaltoPlane(self, original_plane):
        a, b, c, _ = original_plane

        # Arbitrarily choose two components for the new normal vector
        a_prime = 1
        b_prime = 0

        # Solve for c' using the orthogonality condition a * a' + b * b' + c * c' = 0
        c_prime = -(a * a_prime + b * b_prime) / c if c != 0 else 0

        # Define the new perpendicular plane equation: a'x + b'y + c'z + d' = 0
        # d' can be arbitrary
        d_prime = 0
        return np.array([a_prime, b_prime, c_prime, d_prime])

    def fitLine(self, points, max_iterations=1000, threshold=1e-2):
        best_line = None
        best_inliers_count = 0
        best_inliers_mask = None

        intensity_points = np.mean(points, axis=1)
        intensity_points_argsorted = np.argsort(intensity_points)

        bright_pts = intensity_points_argsorted[0 : int(len(intensity_points) * 0.05)]
        dim_pts = intensity_points_argsorted[
            int(len(intensity_points) * 0.95) : len(intensity_points)
        ]

        num_points = points.shape[0]

        for _ in tqdm(range(max_iterations), desc="Estimating best line using RANSAC"):
            # Randomly select 2 points to define the line
            # sample_indices = np.random.choice(num_points, 2, replace=False)
            # p1, p2 = points[sample_indices]

            p1 = points[np.random.choice(dim_pts)]
            p2 = points[np.random.choice(bright_pts)]

            # Compute the direction vector of the line
            direction_vector = p2 - p1
            direction_vector = direction_vector / np.linalg.norm(
                direction_vector
            )  # Normalize

            # Define the line in parametric form: P(t) = P0 + t * d
            p0 = p1  # Use the first point as P0

            # Compute the distance of all points to the line
            distances = np.linalg.norm(np.cross(points - p0, direction_vector), axis=1)
            distances /= np.linalg.norm(direction_vector)  # Normalize distance

            # Identify inliers (points within the threshold distance to the line)
            inliers_mask = distances < threshold
            inliers_count = np.sum(inliers_mask)

            # Update the best line if the current one has more inliers
            if inliers_count > best_inliers_count:
                best_line = (p0, direction_vector)
                best_inliers_count = inliers_count
                best_inliers_mask = inliers_mask

        return [p0, p0 + 10 * direction_vector], best_inliers_mask

    def getPlane(self, normal_vector, point_on_plane):
        normal_vector = normal_vector.astype(np.float64)
        point_on_plane = point_on_plane.astype(np.float64)
        a, b, c = normal_vector

        # Point on the plane (x0, y0, z0)
        x0, y0, z0 = point_on_plane

        # Calculate d using the point on the plane
        d = -(a * x0 + b * y0 + c * z0)
        return a, b, c, d

    def moveptswithlimit(self, pts_proj, plane, arr, type_arr):
        idxs = np.where(arr > 255)
        if type_arr == "x":
            distances = (pts_proj[idxs, 0] - 255) / plane[0]
        elif type_arr == "y":
            distances = (pts_proj[idxs, 1] - 255) / plane[1]
        else:
            distances = (pts_proj[idxs, 2] - 255) / plane[2]
        vec = plane[:3]
        pts_proj[idxs] = pts_proj[idxs] - distances.T * vec
        return pts_proj

    def get3DpointsOnPlane(self, points, a, b, c, d):
        # Normalize the normal vector
        normal_magnitude = np.sqrt(a**2 + b**2 + c**2)
        a, b, c = a / normal_magnitude, b / normal_magnitude, c / normal_magnitude
        d /= normal_magnitude

        # Convert points to a NumPy array for vectorized operations
        points = np.array(points)
        x1, y1, z1 = points[:, 0], points[:, 1], points[:, 2]

        # Calculate the distance from the plane for each point
        distances = a * x1 + b * y1 + c * z1 + d

        # Clip large distances (optional)
        # max_distance = 10.0  # Example threshold
        # distances = np.clip(distances, -max_distance, max_distance)

        # Calculate the projected points
        x_proj = x1 - distances * a
        y_proj = y1 - distances * b
        z_proj = z1 - distances * c

        # Combine projected coordinates
        projected_points = np.vstack((x_proj, y_proj, z_proj)).T
        projected_points = self.moveptswithlimit(
            projected_points, [a, b, c, d], x_proj, "x"
        )
        projected_points = self.moveptswithlimit(
            projected_points, [a, b, c, d], y_proj, "y"
        )
        projected_points = self.moveptswithlimit(
            projected_points, [a, b, c, d], z_proj, "z"
        )

        return projected_points

    def removespots(
        self,
        image,
        mask_img,
        kernel_size=(3, 3),
        iterations=5,
        threshold_ratio=0.5,
        threshold_for_intensity_diff=1.25,
    ):
        image = image.astype(np.float32)

        padded_image = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode="reflect")
        if mask_img is not None:
            mask_img = np.pad(mask_img, ((1, 1), (1, 1)), mode="reflect")

        height, width, _ = image.shape

        for _ in tqdm(range(iterations), desc="Removing inconsistent spots"):
            output_image = np.zeros_like(image, dtype=np.float32)

            intensity = np.mean(padded_image, axis=2)
            for i in range(1, height + 1):  # Adjust for padding
                for j in range(1, width + 1):
                    if mask_img is not None and not mask_img[i, j]:
                        output_image[i - 1, j - 1, :] = padded_image[i, j, :]
                        continue
                    # Extract the 3x3 region around the pixel
                    region = padded_image[i - 1 : i + 2, j - 1 : j + 2, :]
                    region_intensity = intensity[i - 1 : i + 2, j - 1 : j + 2]

                    current_intensity = intensity[i, j]

                    higher_intensity_count = np.sum(
                        region_intensity
                        > threshold_for_intensity_diff * current_intensity
                    )

                    higher_values = region_intensity[
                        region_intensity
                        > threshold_for_intensity_diff * current_intensity
                    ]

                    if higher_intensity_count >= threshold_ratio * (
                        kernel_size[0] * kernel_size[1] - 1
                    ):
                        # max_intensity_idx = np.unravel_index(
                        #     np.argmax(region_intensity), region_intensity.shape
                        # )
                        # output_image[i - 1, j - 1, :] = region[
                        #     max_intensity_idx[0], max_intensity_idx[1], :
                        # ]

                        average_value = np.mean(higher_values)
                        output_image[i - 1, j - 1, :] = average_value
                    # elif current_intensity == np.max(region_intensity):
                    #     # Set all surrounding pixels to the value of the middle pixel
                    #     output_image[i - 1 : i + 2, j - 1 : j + 2, :] = padded_image[
                    #         i, j, :
                    #     ]
                    else:
                        output_image[i - 1, j - 1, :] = padded_image[i, j, :]

            padded_image[1:-1, 1:-1, :] = output_image
        output_image = np.clip(output_image, 0, 255).astype(np.uint8)

        return output_image

    #################################################################3
    #################################################################3
    #################################################################3
    #################################################################3

    # def normalize_kernel(kernel):
    #     total = np.sum(kernel)
    #     if total == 0:
    #         return kernel
    #     return kernel / total

    # def adjust_processed_kernel(original_kernel, processed_kernel):
    #     # Normalize both kernels
    #     norm_original = normalize_kernel(original_kernel)
    #     norm_processed = normalize_kernel(processed_kernel)

    #     # Calculate the ratio between the processed and original kernels
    #     ratio = np.divide(
    #         norm_processed,
    #         norm_original,
    #         out=np.ones_like(norm_processed),
    #         where=norm_original != 0,
    #     )

    #     # If all values are close to 1, intensity is consistent
    #     if np.allclose(ratio, 1, atol=0.10):  # Tolerance set to 10%
    #         return processed_kernel
    #     else:
    #         adjustment = 1 / ratio
    #         adjusted_kernel = processed_kernel * adjustment
    #         print(f"Adjustment factor:\n{adjustment}")
    #         return adjusted_kernel

    def normalize_kernel(self, kernel):
        total = np.sum(kernel)
        if total == 0:
            return kernel
        return kernel / total

    def adjust_processed_kernel(self, original_kernel, processed_kernel):
        # Normalize both kernels
        norm_original = self.normalize_kernel(original_kernel)
        norm_processed = self.normalize_kernel(processed_kernel)

        # Calculate the ratio between the processed and original kernels
        ratio = np.divide(
            norm_processed,
            norm_original,
            out=np.ones_like(norm_processed),
            where=norm_original != 0,
        )

        return ratio

    def calculate_ratio_value_map(self, original_image, processed_image, kernel_size=3):

        original_image = np.mean(original_image, axis=2)
        processed_image = np.mean(processed_image, axis=2)

        # Initialize the ratio value map with -1
        ratio_value_map = np.full(original_image.shape, -1.0)

        pad_size = kernel_size // 2

        # Pad the images to handle borders
        padded_original = np.pad(original_image, pad_size, mode="edge")
        padded_processed = np.pad(processed_image, pad_size, mode="edge")

        # Iterate over each pixel in the original image
        for i in range(original_image.shape[0]):
            for j in range(original_image.shape[1]):
                # Extract the 3x3 kernels
                original_kernel = padded_original[
                    i : i + kernel_size, j : j + kernel_size
                ]
                processed_kernel = padded_processed[
                    i : i + kernel_size, j : j + kernel_size
                ]

                # Calculate the ratio for the current kernel
                ratio = self.adjust_processed_kernel(original_kernel, processed_kernel)

                # # Update the ratio value map
                # if ratio_value_map[i, j] == -1:  # If uninitialized
                #     ratio_value_map[i, j] = ratio.mean()  # Initialize with mean
                # else:
                #     ratio_value_map[i, j] = (
                #         ratio_value_map[i, j] + ratio.mean()
                #     ) / 2  # Average with mean

                # Calculate the average with the existing value in the ratio value map
                for k in range(kernel_size):
                    for l in range(kernel_size):
                        ratio_value = ratio[k, l]
                        if ratio_value_map[i, j] == -1:  # If uninitialized
                            ratio_value_map[i, j] = ratio_value
                        else:
                            # Average the new ratio value with the old one
                            ratio_value_map[i, j] = (
                                ratio_value_map[i, j] + ratio_value
                            ) / 2

        return ratio_value_map

    def apply_blur_with_mask(
        self, image, mask, kernel_size=(3, 3), iterations=3, sigmaX=1
    ):
        mask = (mask / 255).astype(np.uint8)
        # cv2.imshow("before", image)
        # Apply Gaussian blur to the entire image
        blurred_image = image.copy()
        for _ in range(iterations):
            blurred_image = cv2.GaussianBlur(blurred_image, kernel_size, sigmaX)

        # Expand binary mask to 3 channels (so it matches the image dimensions)
        binary_mask_3channel = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Combine the original image and blurred image using the mask
        result = (
            image * (1 - binary_mask_3channel) + blurred_image * binary_mask_3channel
        )

        # Convert the result back to uint8
        result = result.astype(np.uint8)
        # cv2.imshow("result", result)
        # cv2.imshow("hehe", (blurred_image * binary_mask_3channel).astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return result

    def match_intensity(self, rgb_image, gray_image):
        rgb_image = rgb_image.astype(np.float32)
        gray_image = gray_image.astype(np.float32)

        # Calculate the intensity of each pixel in the RGB image (sum of R, G, B)
        rgb_intensity = np.mean(rgb_image, axis=2)  # H x W array

        # Avoid division by zero (handle black pixels)
        mask = rgb_intensity != 0
        scaling_factor = np.zeros_like(rgb_intensity)
        scaling_factor[mask] = gray_image[mask] / rgb_intensity[mask]

        # Adjust the RGB values by scaling each channel proportionally
        adjusted_rgb_image = np.zeros_like(rgb_image)
        for c in range(3):  # For each channel (R, G, B)
            adjusted_rgb_image[:, :, c] = rgb_image[:, :, c] * scaling_factor

        # Clip the values to ensure they are within valid range [0, 255]
        adjusted_rgb_image = np.clip(adjusted_rgb_image, 0, 255).astype(np.uint8)

        return adjusted_rgb_image

    def set_gray_img_as_brightness(self, rgb_image, gray_img, brightness_factor=1.2):
        # Convert the RGB image to HSV color space
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

        # Extract the HSV channels
        h, s, v = cv2.split(hsv_image)

        # Increase the Value (brightness) channel
        # v = gray_img.copy()
        v = np.clip(gray_img * brightness_factor, 0, 255).astype(np.uint8)

        # Merge the HSV channels back
        hsv_brightened = cv2.merge([h, s, v])

        # Convert back to RGB color space
        brightened_rgb_image = cv2.cvtColor(hsv_brightened, cv2.COLOR_HSV2BGR)

        return brightened_rgb_image

    def set_pixels_based_on_xy_map(self, xy_map, rgb_image, outlier_removal=False):
        result_image = rgb_image.copy()

        H, W, _ = rgb_image.shape

        xy_groups = {}

        for i in range(H):
            for j in range(W):
                xy = tuple(xy_map[i, j, :])
                if xy not in xy_groups:
                    xy_groups[xy] = []
                xy_groups[xy].append((i, j))

        for xy, pixel_locations in xy_groups.items():
            rgb_values = np.array([rgb_image[i, j] for i, j in pixel_locations])
            if outlier_removal:
                rgb_values = self.remove_outliers(rgb_values)

            brightness = np.sum(rgb_values, axis=1)

            brightest_idx = np.argmax(brightness)

            brightest_pixel = rgb_values[brightest_idx]
            for i, j in pixel_locations:
                result_image[i, j] = brightest_pixel

        return result_image

    def remove_outliers(self, points, threshold=1):
        orig_pts = points.copy()
        # points = (points - np.min(points)) / (np.max(points) - np.min(points))

        A = points[:, 0]
        B = points[:, 1]
        C = points[:, 2]

        ratios = np.zeros((points.shape[0], 2))
        ratios[:, 0] = np.where(B != 0, A / B, 0)  # A/B
        ratios[:, 1] = np.where(B != 0, C / B, 0)  # C/B

        z_scores = np.abs(stats.zscore(ratios))

        valid_indices = np.where((z_scores < threshold).all(axis=1))[0]

        if len(valid_indices) == 0:
            filtered_points = orig_pts
        else:
            filtered_points = orig_pts[valid_indices]

        return filtered_points

    def getfromlog(self, xy_map, log_img, ratio):
        result_image = log_img.copy()
        H, W, _ = log_img.shape

        xy_groups = {}

        for i in range(H):
            for j in range(W):
                xy = tuple(xy_map[i, j, :])
                if xy not in xy_groups:
                    xy_groups[xy] = []
                xy_groups[xy].append((i, j))

        for xy, pixel_locations in xy_groups.items():
            log_rgb_values = np.array([log_img[i, j] for i, j in pixel_locations])
            log_rgb_values = self.remove_outliers(log_rgb_values)

            brightness = np.sum(log_rgb_values, axis=1)

            brightest_idx = np.argmax(brightness)

            brightest_pixel = log_rgb_values[brightest_idx]
            for i, j in pixel_locations:
                result_image[i, j] = brightest_pixel
        return result_image

    def patchify(self, img, patch_size=128, stride=64):
        """Split the image into patches with the given patch size and stride (overlap)."""
        patches = []
        img_h, img_w = img.shape[0], img.shape[1]
        patch_h, patch_w = patch_size, patch_size

        # Loop over the image with the given stride to create overlapping patches
        for i in range(0, img_h - patch_h + 1, stride):
            for j in range(0, img_w - patch_w + 1, stride):
                patch = img[i : i + patch_h, j : j + patch_w, :]
                patches.append(patch)

        return patches, (img_h, img_w)

    def restitch(self, patches, original_shape, patch_size=128, stride=64):
        """Reassemble patches into the original image with averaging for overlapping regions."""
        img_h, img_w = original_shape
        patch_h, patch_w = patch_size, patch_size

        # Initialize arrays for the reassembled image and a counter for overlap averaging
        reassembled_img = np.zeros(
            (img_h, img_w, patches[0].shape[2]), dtype=np.float32
        )  # Use float32 for reassembled image
        count_matrix = np.zeros((img_h, img_w, 1), dtype=np.float32)

        patch_idx = 0

        # Loop over the image to place patches at their correct locations
        for i in range(0, img_h - patch_h + 1, stride):
            for j in range(0, img_w - patch_w + 1, stride):
                reassembled_img[i : i + patch_h, j : j + patch_w, :] += patches[
                    patch_idx
                ]
                count_matrix[i : i + patch_h, j : j + patch_w, :] += 1
                patch_idx += 1

        # Avoid division by zero and normalize the reassembled image by the count matrix
        reassembled_img /= np.maximum(count_matrix, 1)

        # Convert back to uint8 if needed (optional)
        reassembled_img = np.clip(reassembled_img, 0, 255).astype(np.uint8)

        return reassembled_img

    def convertlogtolinear(self, log_img):
        log_img = log_img.astype(np.float32)
        log_img = np.exp(log_img)
        log_img = cv2.normalize(log_img, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        log_img = log_img.astype(np.uint8)
        return log_img

    def calculate_sobel_magnitude_and_direction(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        direction = np.arctan2(sobel_y, sobel_x)
        direction = np.degrees(direction)
        cv2.imshow("magnitude", magnitude.astype(np.uint8))
        cv2.imshow("direction", direction.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return magnitude, direction

    def getABC(self, grad_dir, img, row_col):
        y, x = row_col
        if (0 <= grad_dir < 22.5) or (157.5 <= grad_dir <= 180):
            # Horizontal gradient - perpendicular points are above and below
            a = img[y - 1, x]
            b = img[y, x]
            c = img[y + 1, x]

        elif 22.5 <= grad_dir < 67.5:
            # 45-degree gradient - perpendicular points are on the other diagonal
            a = img[y - 1, x - 1]
            b = img[y, x]
            c = img[y + 1, x + 1]

        elif 67.5 <= grad_dir < 112.5:
            # Vertical gradient - perpendicular points are left and right
            a = img[y, x - 1]
            b = img[y, x]
            c = img[y, x + 1]

        elif 112.5 <= grad_dir < 157.5:
            # 135-degree gradient - perpendicular points are on the opposite diagonal
            a = img[y - 1, x + 1]
            b = img[y, x]
            c = img[y + 1, x - 1]

        else:
            a, b, c = None, None, None  # Skip if not a defined direction
        return a, b, c

    def process_image(
        self, shadowimage, processed_image, magnitude_threshold=150, iterations=3
    ):
        magnitude_shad, direction_shad = self.calculate_sobel_magnitude_and_direction(
            shadowimage
        )
        shadowimage = cv2.cvtColor(shadowimage, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

        for _ in range(iterations):
            # Apply a threshold to the magnitude map
            strong_edges = magnitude_shad > magnitude_threshold

            # Loop through the magnitude map
            for y in range(1, magnitude_shad.shape[0] - 1):
                for x in range(1, magnitude_shad.shape[1] - 1):
                    if strong_edges[y, x]:
                        grad_dir = direction_shad[y, x]
                        grad_dir = abs(grad_dir)

                        a1, b1, c1 = self.getABC(grad_dir, shadowimage, (y, x))
                        a2, b2, c2 = self.getABC(grad_dir, processed_image, (y, x))

                        # Calculate the ratio a/b and c/b
                        val1, val2 = 0, 0
                        if a1 != 0:  # To avoid division by zero
                            val1 = (a2 * b1) / a1
                        if c1 != 0:  # To avoid division by zero
                            val2 = (c2 * b1) / c1
                        # if (val1 + val2) / 2 > processed_image[y, x]:
                        # print("in here")
                        processed_image[y, x] = (val1 + val2) / 2
                    else:
                        processed_image[y, x] = 0
        cv2.imshow("processed_image", processed_image * 255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()

    def plot_two_sets_of_points(
        self, points1, points2, imname, rgbimg, color1="blue", color2="red"
    ):
        ## NORMALIZATION
        min_val = np.min(points1, axis=0)
        max_val = np.max(points1, axis=0)

        # zero_ch = np.zeros((480, 640, 1), dtype=np.float32)

        points1 = (points1 - min_val) / (max_val - min_val)
        points1 = np.clip(points1, a_min=0, a_max=1)
        points1 = points1.reshape((480, 640, 2))
        # save_path = f"/home/satviktyagi/Desktop/desk/project/logchromaticity_guidance/logchromaticity/fulldataset_vanilla/train/input/{imname}.exr"
        # imageio.imwrite(
        #     save_path, np.concatenate([points1, zero_ch], axis=2), format="EXR"
        # )

        points2 = (points2 - min_val) / (max_val - min_val)
        points2 = np.clip(points2, a_min=0, a_max=1)
        points2 = points2.reshape((480, 640, 2))
        # save_path = save_path.replace("input", "gt")
        # imageio.imwrite(
        #     save_path, np.concatenate([points2, zero_ch], axis=2), format="EXR"
        # )
        gray_points1 = ((np.mean(points1, axis=2)) * 255).astype(np.uint8)
        gray_points2 = ((np.mean(points2, axis=2)) * 255).astype(np.uint8)

        combined = np.hstack([gray_points1, gray_points2])

        # cv2.imshow(f"{imname}_comparison.png", combined)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return

        # cv2.imwrite(
        #     f"/home/satviktyagi/Desktop/desk/project/logchromaticity_guidance/logchromaticity/fulldataset_vanilla/gray/{imname}.png",
        #     combined,
        # )

        x1, y1 = points1[:, 0], points1[:, 1]
        x2, y2 = points2[:, 0], points2[:, 1]

        # Plot each set of points
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(x1, y1, color=color1, label="Input XY map Points")
        plt.scatter(x2, y2, color=color2, label="Target XY map Points")

        # Label axes and add a legend
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.title("Two Sets of 2D Points")
        # plt.show()

        plt.savefig(
            f"/home/satviktyagi/Desktop/desk/project/to_show/for_doc/{imname}_wrong.png",
            dpi=fig.dpi,
        )
        plt.close(fig)

        ##########################################################

        x1, y1 = points1[:, 0], points1[:, 1]
        x2, y2 = points2_correct[:, 0], points2_correct[:, 1]

        # Plot each set of points
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(x1, y1, color=color1, label="Input XY map Points")
        plt.scatter(x2, y2, color=color2, label="Target XY map Points")

        # Label axes and add a legend
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.title("Two Sets of 2D Points")
        # plt.show()

        plt.savefig(
            f"/home/satviktyagi/Desktop/desk/project/to_show/for_doc/{imname}_right.png",
            dpi=fig.dpi,
        )
        plt.close(fig)

        # return points1, points2, gray_points1, gray_points2

    # def rotate_plane_along_y(self, normal_vector, angle_deg):
    #     angle_rad = np.radians(angle_deg)

    #     rotation_matrix_y = np.array(
    #         [
    #             [np.cos(angle_rad), 0, np.sin(angle_rad)],
    #             [0, 1, 0],
    #             [-np.sin(angle_rad), 0, np.cos(angle_rad)],
    #         ]
    #     )
    #     rotated_normal = np.dot(rotation_matrix_y, normal_vector)
    #     return rotated_normal

    # def getRotatedISD(self, ISD_vec, angle):
    #     return self.rotate_plane_along_y(ISD_vec, angle)

    def getmultipleISDs(
        self, log_img, lit_shadow_pts, num_vectors=10, min_neighbor_dist=0.05
    ):
        new_vectors = []
        lit_pt, shadow_pt = lit_shadow_pts[0], lit_shadow_pts[1]

        points = log_img.reshape(-1, 3)
        tree = KDTree(points)

        indices_lit = tree.query_ball_point(lit_pt, min_neighbor_dist)
        indices_shadow = tree.query_ball_point(shadow_pt, min_neighbor_dist)

        random.shuffle(indices_lit)
        random.shuffle(indices_shadow)

        min_indices = min(len(indices_lit), len(indices_shadow))

        for i in range(min_indices):
            new_vectors.append(points[indices_lit[i]] - points[indices_shadow[i]])
            if i + 1 == num_vectors:
                break

        return new_vectors
