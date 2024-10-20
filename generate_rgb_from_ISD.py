import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
import re
import random
import math

from logchromaticity import processLogImage
from utils import Image

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
PATH_RGB = f"/home/satviktyagi/Desktop/desk/project/datasets/des3_dataset/ISTD_des3/train/train_A/"
PATH_MASK = f"/home/satviktyagi/Desktop/desk/project/datasets/des3_dataset/ISTD_des3/train/train_B/"
PATH_LOG = f"/home/satviktyagi/Desktop/desk/project/datasets/des3_dataset/ISTD_pseduo_log/pseudolog/train/train_A/"


def parse_args():
    parser = argparse.ArgumentParser(description="Creating Log Chromaticity Image")
    parser.add_argument(
        "--ISDpath",
        default=None,
        type=str,
        help="text file path ISD points",
    )
    parser.add_argument(
        "--savepath",
        default=None,
        type=str,
        help="path to save RGB",
    )
    parser.add_argument(
        "--random",
        default=None,
        type=str,
        help="shuffle the images",
    )
    parser.add_argument(
        "--start_from",
        default=0,
        type=int,
        help="start from image number...",
    )
    parser.add_argument(
        "--usemask",
        default=None,
        type=str,
        help="flag to use mask or not",
    )
    args = parser.parse_args()
    return args


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


def main():
    used_image_set = set()
    args = parse_args()
    Image_ = Image()
    logproc = processLogImage()

    data_dict = list(parse_file(args.ISDpath).items())

    if args.random:
        random.shuffle(data_dict)

    for idx in tqdm(
        range(args.start_from, len(data_dict)), desc="Reading log and RGB images"
    ):

        while args.random and True:
            idx = random.choice(range(args.start_from, len(data_dict)))
            if idx not in used_image_set:
                used_image_set.add(idx)
                break

        imgname, [lit_pt, shadow_pt] = data_dict[idx]

        log_img = cv2.imread(
            os.path.join(PATH_LOG, f"{imgname}.exr"), cv2.IMREAD_UNCHANGED
        )
        # log_img = cv2.GaussianBlur(log_img, (5, 5), 1.0)

        rgb_img = cv2.imread(
            os.path.join(PATH_RGB, f"{imgname}.png"), cv2.IMREAD_UNCHANGED
        )

        mask_img, mask_pts = None, None
        if args.usemask:
            mask_img = cv2.imread(
                os.path.join(PATH_MASK, f"{imgname}.png"), cv2.IMREAD_GRAYSCALE
            ).astype(bool)
            mask_pts = mask_img.reshape(-1, 1)

        points = [lit_pt, shadow_pt]
        lit_shadow_pts = [
            log_img[points[-2][0], points[-2][1], :],
            log_img[points[-1][0], points[-1][1], :],
        ]
        ISD_vec = lit_shadow_pts[0] - lit_shadow_pts[1]

        tfmat = logproc.estimateTransformationMat(ISD_vec)
        projected_points_2d = logproc.transformPoints(log_img, tfmat)
        gray_img = logproc.estimateGrayscaleImagefromXY(
            projected_points_2d, log_img.shape
        )

        projected_points_2d_new = logproc.plot2Dpts(
            projected_points_2d, threshold=0.05, show_plot=False
        )
        gray_img_new = logproc.estimateGrayscaleImagefromXY(
            projected_points_2d_new, log_img.shape
        )

        # cv2.imshow("gray_img", gray_img)
        # cv2.imshow("gray_img_new", gray_img_new)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # continue
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        gray_img_new = cv2.cvtColor(gray_img_new, cv2.COLOR_GRAY2BGR)

        projected_points_2d = projected_points_2d_new
        # Image_.showImage(gray_img)

        # rgbimg = logproc.reprojectXYtoRGB(projected_points_2d, constant_intensity=0)
        # rgbimg = rgbimg.reshape(log_img.shape).astype(np.uint8)
        # Image_.showImage(rgbimg)

        ##############################################################################
        # print("Calculate color image or not 1 or 2")
        # choice = input()
        # if choice != "1":
        #     continue
        point_color_dict = {}
        min_x, max_x = np.min(projected_points_2d[:, 0]), np.max(
            projected_points_2d[:, 0]
        )
        min_y, max_y = np.min(projected_points_2d[:, 1]), np.max(
            projected_points_2d[:, 1]
        )

        for i, (point_2d, point_3d) in tqdm(
            enumerate(zip(projected_points_2d, rgb_img.reshape(-1, 3))),
            desc="Retreiving Original Colors",
        ):
            x, y = point_2d[0], point_2d[1]
            x_normalized = int(((x - min_x) / (max_x - min_x)) * 255)
            y_normalized = int(((y - min_y) / (max_y - min_y)) * 255)
            norm_point_2d = (x_normalized, y_normalized)
            tuple_pt_3d = tuple((np.mean(point_3d), point_3d))
            if norm_point_2d not in point_color_dict:
                point_color_dict[norm_point_2d] = [tuple_pt_3d, 1]
            elif tuple_pt_3d[0] > point_color_dict[norm_point_2d][0][0]:
                point_color_dict[norm_point_2d][0] = tuple_pt_3d
                point_color_dict[norm_point_2d][1] += 1
            else:
                point_color_dict[norm_point_2d][1] += 1

        projected_points_RGB = logproc.reprojectXYtoOriginalColor(
            projected_points_2d,
            rgb_img.reshape(-1, 3),
            mask_pts,
            point_color_dict,
            minimum_same_mapped=0,
            intensity=None,
        )
        reprojected_rgbimg = projected_points_RGB.reshape(rgb_img.shape)
        # reprojected_rgbimg__ = logproc.removespots(
        #     reprojected_rgbimg,
        #     mask_img,
        #     kernel_size=(3, 3),
        #     iterations=1,
        #     threshold_ratio=0.5,
        #     threshold_for_intensity_diff=1.25,
        # )

        color_img = reprojected_rgbimg.copy()
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        reprojected_rgbimg = logproc.removespots(
            reprojected_rgbimg,
            mask_img,
            kernel_size=(3, 3),
            iterations=2,
            threshold_ratio=0.5,
            threshold_for_intensity_diff=1.25,
        )

        iterations = 3
        for i in tqdm(range(iterations), desc="Applying ratio map"):
            ratio_map = logproc.calculate_ratio_value_map(
                rgb_img, reprojected_rgbimg, kernel_size=3
            )
            ratio_map_expanded = ratio_map[
                :, :, np.newaxis
            ]  # Expand dims to (height, width, 1)

            reprojected_rgbimg = np.multiply(
                reprojected_rgbimg.astype(np.float32), ratio_map_expanded
            )

        reprojected_rgbimg = np.clip(reprojected_rgbimg, 0, 255).astype(np.uint8)

        # reprojected_rgbimg = logproc.removespots(
        #     reprojected_rgbimg,
        #     mask_img,
        #     kernel_size=(3, 3),
        #     iterations=1,
        #     threshold_ratio=0.5,
        #     threshold_for_intensity_diff=1.25,
        # )

        # cv2.imshow("original", rgb_img)
        # cv2.imshow("gray_img", gray_img)
        # cv2.imshow("reprojected", color_img)
        # cv2.imshow("ratiomapimg", reprojected_rgbimg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        combined_img = np.hstack([rgb_img, gray_img_new, color_img, reprojected_rgbimg])
        cv2.imwrite(
            f"performance_comparison/{idx}_withGrayMapChanges.png", combined_img
        )
        continue

        # exit()

        ##############################################################################

        # # max_log_intensity_idx = np.argmax(np.sum(log_img.reshape(-1, 3), axis=1))
        # # max_log_intensity_val = log_img.reshape(-1, 3)[max_log_intensity_idx, :]

        # # a_init, b_init, c_init, d_init = logproc.getPlane(ISD_vec, lit_shadow_pts[0])

        # lit_shadow_pts[0] = lit_shadow_pts[0] + math.log(2) * ISD_vec
        # ISD_vec = lit_shadow_pts[0] - lit_shadow_pts[1]

        # a, b, c, d = logproc.getPlane(ISD_vec, lit_shadow_pts[0])

        # projected_3D_pts = logproc.get3DpointsOnPlane(
        #     log_img.reshape(-1, 3), a, b, c, d
        # )
        # reprojected_logimg = projected_3D_pts.reshape(log_img.shape)

        # logproc.generate3DHistogram(
        #     reprojected_logimg.reshape(-1, 3),
        #     color_image=rgb_img,
        #     save_name=f"reprojlogcloud.html",
        #     title="LOG PCL",
        #     vector=[lit_shadow_pts],
        #     planes=[[a, b, c, d]],
        #     center_point=lit_shadow_pts[0],
        # )

        # reprojected_logimg = Image_.convertlogtolinear(reprojected_logimg)
        # # reprojected_logimg = Image_.setconstantIntensity(
        # #     reprojected_logimg, intensity=100
        # # )
        # cv2.imshow("projlogimg", reprojected_logimg.astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # # logproc.generate3DHistogram(
        # #     log_img.reshape(-1, 3),
        # #     color_image=rgb_img,
        # #     save_name=f"logcloud.html",
        # #     title="LOG PCL",
        # #     vector=[lit_shadow_pts],
        # #     planes=[[a, b, c, d]],
        # #     center_point=lit_shadow_pts[0],
        # # )

        #############################################################################################
        # print("Calculate color image or not 1 or 2")
        # choice = input()
        # if choice != "1":
        #     continue
        # point_color_dict = {}
        # min_x, max_x = np.min(projected_3D_pts[:, 0]), np.max(projected_3D_pts[:, 0])
        # min_y, max_y = np.min(projected_3D_pts[:, 1]), np.max(projected_3D_pts[:, 1])
        # min_z, max_z = np.min(projected_3D_pts[:, 2]), np.max(projected_3D_pts[:, 2])

        # for i, (proj_point_3d, color_3d) in tqdm(
        #     enumerate(zip(projected_3D_pts, rgb_img.reshape(-1, 3))),
        #     desc="Retreiving Original Colors",
        # ):
        #     x, y, z = proj_point_3d[0], proj_point_3d[1], proj_point_3d[2]
        #     x_normalized = int(((x - min_x) / (max_x - min_x)) * 255)
        #     y_normalized = int(((y - min_y) / (max_y - min_y)) * 255)
        #     z_normalized = int(((z - min_z) / (max_z - min_z)) * 255)

        #     norm_point_3d = (x_normalized, y_normalized, z_normalized)
        #     tuple_pt_3d = tuple(color_3d)
        #     if norm_point_3d not in point_color_dict:
        #         point_color_dict[norm_point_3d] = [set([tuple_pt_3d]), 1]
        #     elif tuple_pt_3d not in point_color_dict[norm_point_3d][0]:
        #         point_color_dict[norm_point_3d][0].add(tuple_pt_3d)
        #         point_color_dict[norm_point_3d][1] += 1
        #     else:
        #         point_color_dict[norm_point_3d][1] += 1

        # projected_points_RGB = logproc.temp_hehe(
        #     projected_3D_pts,
        #     rgb_img.reshape(-1, 3),
        #     point_color_dict,
        #     mask_pts=None,
        #     minimum_same_mapped=0,
        #     intensity=None,
        # )
        # reprojected_rgbimg = projected_points_RGB.reshape(rgb_img.shape)
        # # reprojected_rgbimg = logproc.removespots(
        # #     reprojected_rgbimg,
        # #     mask_img,
        # #     kernel_size=(3, 3),
        # #     iterations=2,
        # #     threshold_ratio=0.5,
        # #     threshold_for_intensity_diff=1.25,
        # # )

        # color_img = reprojected_rgbimg
        # cv2.imshow("reprojected", color_img)
        # cv2.imshow("original", rgb_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        ################################################################################################
        # exit()


if __name__ == "__main__":
    main()
