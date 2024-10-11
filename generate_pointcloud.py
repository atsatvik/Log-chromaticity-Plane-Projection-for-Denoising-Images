import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from tqdm import tqdm
import argparse

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def plot_rgb_3d_histogram(
    image, color_image, bin_size=64, save_name="3d_rgb_histogram", title="pcl"
):
    upper_lim = np.max(image)
    lower_lim = np.min(image)

    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    r = r.flatten()
    g = g.flatten()
    b = b.flatten()

    r_color, g_color, b_color = (
        color_image[:, :, 0],
        color_image[:, :, 1],
        color_image[:, :, 2],
    )

    r_color = r_color.flatten()
    g_color = g_color.flatten()
    b_color = b_color.flatten()

    # hist, edges = np.histogramdd(
    #     (r, g, b),
    #     bins=(bin_size, bin_size, bin_size),
    #     range=((lower_lim, upper_lim), (lower_lim, upper_lim), (lower_lim, upper_lim)),
    # )
    # r_bin = np.digitize(r, edges[0]) - 1
    # g_bin = np.digitize(g, edges[1]) - 1
    # b_bin = np.digitize(b, edges[2]) - 1

    # # Clip the bin indices to make sure they are within the valid range
    # r_bin = np.clip(r_bin, 0, bin_size - 1)
    # g_bin = np.clip(g_bin, 0, bin_size - 1)
    # b_bin = np.clip(b_bin, 0, bin_size - 1)

    # x_vals = []
    # y_vals = []
    # z_vals = []
    # sizes = []
    # colors = []

    # bin_edges = np.linspace(0, 255, bin_size + 1)
    # r_bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # g_bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # b_bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # # Access the bin centers using the bin index
    # r_center = r_bin_centers[r_bin]
    # g_center = g_bin_centers[g_bin]
    # b_center = b_bin_centers[b_bin]

    # # Combine the RGB bin centers
    # rgb_corresponding_bin_centers = np.stack([r_center, g_center, b_center], axis=1)

    # for r_idx in range(bin_size):
    #     for g_idx in range(bin_size):
    #         for b_idx in range(bin_size):
    #             if hist[r_idx, g_idx, b_idx] > 0:
    #                 x_vals.append(r_bin_centers[r_idx])
    #                 y_vals.append(g_bin_centers[g_idx])
    #                 z_vals.append(b_bin_centers[b_idx])
    #                 if hist[r_idx, g_idx, b_idx] > 20:
    #                     size = 10
    #                 else:
    #                     size = hist[r_idx, g_idx, b_idx]
    #                 sizes.append(size)
    #                 colors.append(
    #                     (
    #                         r_bin_centers[r_idx],
    #                         g_bin_centers[g_idx],
    #                         b_bin_centers[b_idx],
    #                     )
    #                 )

    # x_vals, y_vals, z_vals = filter_points_with_neighbors(
    #     x_vals, y_vals, z_vals, threshold=5, min_neighbors=5
    # )

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

    fig.update_layout(
        scene=dict(
            xaxis_title="Red",
            yaxis_title="Green",
            zaxis_title="Blue",
        ),
        title=title,
    )

    fig.write_html(save_name)
    return x_vals, y_vals, z_vals, None


def parse_args():
    parser = argparse.ArgumentParser(description="Creating Log Chromaticity Image")
    parser.add_argument(
        "--img",
        default=None,
        type=str,
        help="name (if exist in same dir) or path of image to process",
    )
    args = parser.parse_args()
    return args


def filter_points_with_neighbors(
    x_vals, y_vals, z_vals, threshold=1.0, min_neighbors=5
):
    # Step 1: Combine x, y, z values into 3D points
    points = np.array(list(zip(x_vals, y_vals, z_vals)))

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

    # Step 4: Unpack the valid points into x, y, z lists
    valid_points = np.array(valid_points)
    if len(valid_points) > 0:
        x_vals_filtered = valid_points[:, 0]
        y_vals_filtered = valid_points[:, 1]
        z_vals_filtered = valid_points[:, 2]
    else:
        x_vals_filtered, y_vals_filtered, z_vals_filtered = [], [], []

    return x_vals_filtered, y_vals_filtered, z_vals_filtered


def main():
    args = parse_args()
    IMAGENAME = args.img
    IMAGENAME = IMAGENAME.split(".")[0]

    path = f"/home/satviktyagi/Desktop/desk/project/datasets/des3_dataset/ISTD_des3/test/test_A/{IMAGENAME}.png"
    rgbimg = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    path = f"{IMAGENAME}.exr"
    image_log = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    plot_rgb_3d_histogram(
        image_log, rgbimg, save_name="pcl_logrgb.html", title="Log PCL"
    )
    plot_rgb_3d_histogram(rgbimg, rgbimg, save_name="pcl_srgb.html", title="RGB PCL")


if __name__ == "__main__":
    main()
