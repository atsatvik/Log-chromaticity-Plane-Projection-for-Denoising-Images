import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
import re
import random
import math
import pandas as pd
from scipy.spatial import KDTree
import scipy.stats
import cv2


def parse_file(file_path):
    data_dict = {}
    with open(file_path, "r") as file:
        for line in file:
            match = re.match(
                r"(\S+)\s+lit_pt, shadow_pt\s+:\s+\((\d+),\s*(\d+)\)\s+\((\d+),\s*(\d+)\)",
                line,
            )
            if match:
                # Extract the image name and points
                image_name = match.group(1)
                lit_pt = (int(match.group(2)), int(match.group(3)))
                shadow_pt = (int(match.group(4)), int(match.group(5)))

                # Store in the dictionary
                data_dict[image_name] = [lit_pt, shadow_pt]
    return data_dict


def convertlogtolinear(log_img):
    log_img = log_img.astype(np.float32)
    log_img = np.exp(log_img)
    log_img = cv2.normalize(log_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return log_img


def read_csv(path_csv):
    data = pd.read_csv(path_csv)
    names = data["filename"]
    ISD_vecs = [data["log_sr_r"], data["log_sr_g"], data["log_sr_b"]]
    ISD_vec_dict = {}
    for i, name in enumerate(names):
        name = name.split(".")[0]
        ISD_vec_dict[name] = np.array([ISD_vecs[2][i], ISD_vecs[1][i], ISD_vecs[0][i]])
    return ISD_vec_dict


def getShadowPt(log_img, threshold=0.1):
    intensity = np.mean(log_img, axis=2)
    intensity_flat = intensity.reshape(-1)

    sorted_indices = np.argsort(intensity_flat)
    # sorted_indices = sorted_indices[len(sorted_indices) // 2 :]

    point_loc = np.unravel_index(sorted_indices[0], (intensity.shape))
    return point_loc


def getRefinedColorImg(rgb_img, gray_image, clahe=False):
    rgb_img = rgb_img.astype(np.uint8)

    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))

    # if clahe:
    #     gray_image = apply_clahe(gray_image)

    lab_planes[0] = gray_image
    lab = cv2.merge(lab_planes)
    rgb_img_final = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return rgb_img_final


def normalize(img, scale=1):
    min_val = np.min(img, axis=(0, 1), keepdims=True)
    max_val = np.max(img, axis=(0, 1), keepdims=True)
    norm_img = (img - min_val) / (max_val - min_val)
    norm_img = norm_img * scale
    return norm_img


def estimateTransformationMat(normal_vector):
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
    v1 = v1 / np.linalg.norm(v1)
    # Find another orthogonal vector to both normal_vector and v1 using cross product
    v2 = np.cross(normal_vector, v1)
    v2 = v2 / np.linalg.norm(v2)

    transformation_matrix = np.vstack([v1, v2])
    return transformation_matrix


def getXYmap(image, tf_mat):
    image = image.reshape(-1, 3)
    projected_points = np.dot(image, tf_mat.T)
    return projected_points


def transform3Dpoints(img, transformation_matrix):
    points = img.reshape(-1, 3)
    # Convert points to homogeneous coordinates (N, 4)
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])

    # Apply the transformation matrix
    transformed_homogeneous = np.dot(transformation_matrix, points_homogeneous.T).T

    # Convert back to 3D coordinates by dropping the homogeneous coordinate
    transformed_points = (
        transformed_homogeneous[:, :3] / transformed_homogeneous[:, 3][:, np.newaxis]
    )

    return transformed_points.reshape(img.shape)


def get3Dtfmat(a, b, c, d):
    normal = np.array([a, b, c])
    normal_norm = normal / np.linalg.norm(normal)  # Normalize

    if a != 0:
        point_on_plane = np.array([-d / a, 0, 0])
    elif b != 0:
        point_on_plane = np.array([0, -d / b, 0])
    elif c != 0:
        point_on_plane = np.array([0, 0, -d / c])
    else:
        raise ValueError("Invalid plane equation coefficients.")

    target = np.array([0, 0, 1])

    v = np.cross(normal_norm, target)
    s = np.linalg.norm(v)
    c = np.dot(normal_norm, target)

    if s != 0:  # Handle the case where the vectors are already aligned
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + np.dot(K, K) * ((1 - c) / (s**2))
    else:
        R = np.eye(3)  # No rotation needed

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = -np.dot(R, point_on_plane)

    new_normal = np.dot(R, normal)
    plane_pt = np.append(point_on_plane, 1)
    plane_pt = np.dot(transformation_matrix, plane_pt)[:3]
    new_d = -np.dot(new_normal, plane_pt)

    return transformation_matrix, (*new_normal, new_d)


def resize_imgs(img_ls, size):
    for i in range(len(img_ls)):
        img_ls[i] = cv2.resize(img_ls[i], size)
    return img_ls


def random_shuffle_dict(ISD_vec_dict):
    items = list(ISD_vec_dict.items())
    random.shuffle(items)
    new_dict = {}
    for k, v in items:
        new_dict[k] = v
    return new_dict


def gethandpickedISD(log_img, points):
    lit_shadow_pts = [
        log_img[points[-2][0], points[-2][1], :],
        log_img[points[-1][0], points[-1][1], :],
    ]
    ISD_vec = lit_shadow_pts[0] - lit_shadow_pts[1]
    return ISD_vec, lit_shadow_pts
