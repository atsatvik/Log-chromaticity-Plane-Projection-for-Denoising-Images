import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import cv2
import numpy as np
from tqdm import tqdm
import re
import random
import math
import pandas as pd
from scipy.spatial import KDTree
import scipy.stats
import cv2

from PCLdebugutils.helper import *
from utils.logchromaticity import processLogImage

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description="Creating Log Chromaticity Image")
    parser.add_argument(
        "--handpickedISD",
        default=None,
        type=str,
        help="Use hand picked ISDs",
    )
    parser.add_argument(
        "--networkISD",
        default=None,
        type=str,
        help="Use spectral ratio network ISDs",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    logpth = "/home/satviktyagi/Desktop/desk/project/logchromaticity_guidance/ISTD/log_data/train/input"  # path to log images
    rgbpth = "/home/satviktyagi/Desktop/desk/project/datasets/des3_dataset/ISTD_des3/train/train_A"  # path to rgb images

    logimgs = os.listdir(logpth)
    ISD_vec_dict = {img.split(".")[0]: None for img in logimgs}

    logproc = processLogImage()

    # To use ISD_data from spectral ratio network
    if args.networkISD:
        ISD_vec_dict = read_csv("ISDs/ISD_spectralrationet.csv")
        ISD_vec_dict = random_shuffle_dict(ISD_vec_dict)
    # To use hand picked ISDs
    elif args.handpickedISD:
        lit_shadow_pts_dict = parse_file("ISDs/ISD_vec_train.txt")
    else:
        raise Exception("Set either --handpickedISD or --networkISD")

    for imname, ISD_vec in ISD_vec_dict.items():
        # to manually use a specific image
        # imname = "34-1"

        log_path = os.path.join(logpth, imname + ".exr")
        log_img = cv2.imread(log_path, cv2.IMREAD_UNCHANGED)
        log_img = normalize(log_img, scale=255).astype(np.uint8)

        rgb_path = os.path.join(rgbpth, imname + ".png")
        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)

        mean_pt = np.mean(
            log_img.reshape(-1, 3), axis=0
        )  # point at which the plane will be placed for color reproj

        if args.handpickedISD:
            ISD_vec, lit_shadow_pts = gethandpickedISD(
                log_img, lit_shadow_pts_dict[imname]
            )
        elif args.networkISD:
            lowest_intensity_pt = getShadowPt(log_img)
            lit_shadow_pts = [mean_pt + 50 * ISD_vec, mean_pt]

        ISD_vec = ISD_vec / np.linalg.norm(ISD_vec)
        lit_shadow_ls = [lit_shadow_pts]

        a, b, c, d = logproc.getPlane(ISD_vec, mean_pt)

        planes_ls = [[a, b, c, d]]

        # Get chromaticity map
        log_chroma_init = logproc.get3DpointsOnPlane(log_img.reshape(-1, 3), a, b, c, d)
        log_chroma = log_chroma_init.reshape(log_img.shape).astype(np.uint8)

        # Get intensity map
        tfmat = estimateTransformationMat(ISD_vec)
        xymap = getXYmap(log_img, tfmat).reshape(
            (log_img.shape[0], log_img.shape[1], 2)
        )
        xymap = normalize(xymap, scale=255)
        intensity_map = np.mean(xymap, axis=2).astype(np.uint8)

        # get refined color image by combining the two maps
        color_img = getRefinedColorImg(log_chroma, intensity_map, clahe=False)

        # for visualizing the log img, log chromaticity map, intensity map and reprojected image side by side
        # intensity_map = cv2.cvtColor(intensity_map, cv2.COLOR_GRAY2BGR)
        # combined = np.hstack(
        #     [log_img.astype(np.uint8), log_chroma, intensity_map, color_img]
        # )
        # cv2.imshow("combined", combined)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()

        log_img, log_chroma, rgb_img = resize_imgs(
            [log_img, log_chroma, rgb_img], size=(100, 100)
        )  # resizeing for to save space and processing while visualizing

        logproc.generate3DHistogram(
            log_img.reshape(-1, 3),
            log_chroma_init.reshape(-1, 3),
            color_image=rgb_img,
            save_name=f"pcl.html",
            title="PCL",
            vector=lit_shadow_ls,
            planes=planes_ls,
            center_point=mean_pt,
        )
        print("Press enter to generate PCL for next image")
        input()
