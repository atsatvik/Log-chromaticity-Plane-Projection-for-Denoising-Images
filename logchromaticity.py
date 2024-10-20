import cv2
import numpy as np
import os
import plotly.graph_objects as go
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


class processLogImage:
    def generate3DHistogram(
        self,
        image,
        color_image=None,
        save_name="pcl.html",
        title="pcl",
        vector=None,
        planes=None,
        center_point=None,
    ):
        upper_lim = np.max(image)
        lower_lim = np.min(image)

        r, g, b = image[:, 0], image[:, 1], image[:, 2]
        r = r.flatten()
        g = g.flatten()
        b = b.flatten()

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
        x_vals, y_vals, z_vals = r, g, b
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
            range_size = 3
            for pl in planes:
                a, b, c, d = pl

                if center_point is not None:
                    x0, y0, z0 = center_point
                else:
                    x0, y0, z0 = 0, 0, 0  # Default center point

                # Create a meshgrid centered around the point (x0, y0)
                half_range = range_size / 2
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
                        opacity=0.7,
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

    # def plot2Dpts(self, points):

    #     points_array = np.array(points)

    #     tree = KDTree(points_array)
    #     merged_points = []
    #     visited = set()  # To keep track of visited points
    #     threshold = 0.1
    #     for i, point in enumerate(points_array):
    #         if i in visited:
    #             continue

    #         # Find indices of points within the threshold distance
    #         indices = tree.query_ball_point(point, threshold)

    #         # Merge points by taking their mean
    #         close_points = points_array[indices]
    #         merged_point = close_points.mean(axis=0)

    #         merged_points.append(merged_point)

    #         # Mark all close points as visited
    #         visited.update(indices)

    #     merged_points = np.array(merged_points)
    #     x, y = merged_points[:, 0], merged_points[:, 1]

    #     x = (x - np.min(x)) / (np.max(x) - np.min(x))
    #     y = (y - np.min(y)) / (np.max(y) - np.min(y))

    #     # Create a scatter plot
    #     plt.scatter(x, y, color="blue", marker="o")
    #     plt.title("Scatter Plot of 2D Points")
    #     plt.xlabel("X-axis")
    #     plt.ylabel("Y-axis")
    #     plt.grid()
    #     plt.show()
    #     # return np.array(merged_points)

    def plot2Dpts(self, points, threshold=0.05, show_plot=False):
        points_array = np.array(points)

        tree = KDTree(points_array)
        merged_points = []
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

            merged_points.append(merged_point)

            # Set the same merged point for all close points in the updated array
            for idx in indices:
                updated_points_array[idx] = merged_point

            # Mark all close points as visited
            visited.update(indices)

        merged_points = np.array(merged_points)

        # Normalize the original points for plotting
        x, y = points_array[:, 0], points_array[:, 1]
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))

        # Normalize the updated points for plotting
        updated_x, updated_y = updated_points_array[:, 0], updated_points_array[:, 1]
        updated_x = (updated_x - np.min(updated_x)) / (
            np.max(updated_x) - np.min(updated_x)
        )
        updated_y = (updated_y - np.min(updated_y)) / (
            np.max(updated_y) - np.min(updated_y)
        )

        if show_plot:
            # Plot original and updated points side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Plot the original points
            ax1.scatter(x, y, color="blue", marker="o")
            ax1.set_title("Original Points")
            ax1.set_xlabel("X-axis")
            ax1.set_ylabel("Y-axis")
            ax1.grid()

            # Plot the updated (merged) points
            ax2.scatter(updated_x, updated_y, color="green", marker="o")
            ax2.set_title("Updated Points (After Merging)")
            ax2.set_xlabel("X-axis")
            ax2.set_ylabel("Y-axis")
            ax2.grid()

            plt.show()
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
        print("Vector from Shadow to Lit:", normal_vector)
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

        x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min()) * 255
        y_norm = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min()) * 255

        grayscale_values = 0.5 * (x_norm + y_norm)
        grayscale_image = grayscale_values.astype(np.uint8)

        grayscale_image = grayscale_image.reshape(image_shape[0], image_shape[1])

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

    def filterPointsbyNearestNeighbor(self, points, threshold=1.0, min_neighbors=5):
        valid_points = []

        num_points = len(points)
        for i in range(num_points):
            distances = np.linalg.norm(points - points[i], axis=1)

            # Count how many points are within the threshold (excluding the point itself)
            neighbor_count = (
                np.sum(distances < threshold) - 1
            )  # Subtract 1 to exclude the point itself

            # If the point has at least min_neighbors, add it to the valid points list
            if neighbor_count >= min_neighbors:
                valid_points.append(points[i])

        valid_points = np.array(valid_points)
        return valid_points

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

    def estimateISDwithNeigbors(self, logimage, lit_shadow_pts, kernel=(3, 3)):
        lit_pt = lit_shadow_pts[0]
        shadow_pt = lit_shadow_pts[1]
        lit_pt_logval = self.calculate_kernel_average(
            logimage, lit_pt[0], lit_pt[1], kernel
        )
        shadow_pt_logval = self.calculate_kernel_average(
            logimage, shadow_pt[0], shadow_pt[1], kernel
        )
        ISD_vec = np.array(lit_pt_logval - shadow_pt_logval)
        return ISD_vec

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
        a, b, c = normal_vector

        # Point on the plane (x0, y0, z0)
        x0, y0, z0 = point_on_plane

        # Calculate d using the point on the plane
        d = -(a * x0 + b * y0 + c * z0)

        # Return the plane equation coefficients (a, b, c, d)
        return a, b, c, d

    def get3DpointsOnPlane(self, points, a, b, c, d):
        normal_length_squared = a**2 + b**2 + c**2

        # Convert points to a NumPy array for vectorized operations
        points = np.array(points)

        # Separate the coordinates
        x1, y1, z1 = points[:, 0], points[:, 1], points[:, 2]

        # Calculate the distance from the plane for each point (with custom d)
        distances = (a * x1 + b * y1 + c * z1 + d) / normal_length_squared

        # Calculate the projected points
        x_proj = x1 - distances * a
        y_proj = y1 - distances * b
        z_proj = z1 - distances * c

        # Combine projected coordinates back into a single array
        projected_points = np.vstack((x_proj, y_proj, z_proj)).T

        return projected_points

    def reconstruct_from_gradients(self, grad_x, grad_y, processed_channel):
        """
        Reconstruct the image from gradients using integration.

        :param grad_x: Horizontal gradients.
        :param grad_y: Vertical gradients.
        :param processed_channel: Original processed channel to preserve intensity.

        :return: Reconstructed image with modified gradients.
        """
        # Initialize result with the processed image channel
        result = processed_channel.copy()

        # Integrate gradients (basic approach to integrate gradient fields)
        for i in range(1, grad_x.shape[0]):
            result[i, :] = result[i - 1, :] + grad_y[i, :]

        for j in range(1, grad_x.shape[1]):
            result[:, j] = result[:, j - 1] + grad_x[:, j]

        return result

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


##################################################
# import numpy as np

# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# b = np.array([[2, 4, 6], [8, 10, 12], [1, 3, 7]])
# print("orig_a", a, "\n")
# print("orig_b", b, "\n")

# a_norm = a / np.sum(a)
# b_norm = b / np.sum(b)

# ratio = b_norm / a_norm

# print("a_norm", a_norm, "\n")
# print("b_norm", b_norm, "\n")
# print("ratio", ratio, "\n")

# adjustment = 1 / ratio
# print(adjustment)
# b = b * adjustment


# print("orig a", a, "\n")
# print("adjusted_b", b, "\n")


# print(a / np.sum(a))
# print(b / np.sum(b))
##################################################3
