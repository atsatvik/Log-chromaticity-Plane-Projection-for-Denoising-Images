import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
import re
import random
import math

from utils.logchromaticity import processLogImage
from utils.utils import Image

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
PATH_RGB = f"/home/satviktyagi/Desktop/desk/project/datasets/des3_dataset/ISTD_des3/test/test_A/"
PATH_MASK = f"/home/satviktyagi/Desktop/desk/project/datasets/des3_dataset/ISTD_des3/test/test_B/"
PATH_LOG = f"/home/satviktyagi/Desktop/desk/project/datasets/des3_dataset/ISTD_pseduo_log/pseudolog/test/test_A/"
PATH_GROUNDTRUTH = f"/home/satviktyagi/Desktop/desk/project/datasets/des3_dataset/ISTD_pseduo_log/pseudolog/test/test_C/"


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
    parser.add_argument(
        "--from_name",
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

    if args.from_name:
        for i, elem in enumerate(data_dict):
            if elem[0].split(".")[0] == args.from_name:
                args.start_from = i
                break

    if args.random:
        random.shuffle(data_dict)
    # idx_ls = [
    #     0,
    #     3,
    #     15,
    #     25,
    #     50,
    #     100,
    #     150,
    #     200,
    #     300,
    #     400,
    #     500,
    #     600,
    #     700,
    #     800,
    #     850,
    #     900,
    #     950,
    #     1000,
    #     1050,
    #     1100,
    #     1300,
    # ]
    # for i in range(len(idx_ls)):
    #     if idx_ls[i] + 17 not in idx_ls:
    #         idx_ls.append(idx_ls[i] + 17)

    for idx in tqdm(
        range(args.start_from, len(data_dict)), desc="Reading log and RGB images"
    ):
        # for idx in tqdm(idx_ls, desc="Reading log and RGB images"):
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

        ground_truth = cv2.imread(
            os.path.join(PATH_GROUNDTRUTH, f"{imgname}.exr"), cv2.IMREAD_UNCHANGED
        )

        mask_img, mask_pts = None, None
        if args.usemask:
            mask_img = cv2.imread(
                os.path.join(PATH_MASK, f"{imgname}.png"), cv2.IMREAD_GRAYSCALE
            ).astype(bool)
            mask_pts = mask_img.reshape(-1, 1)

        # mask_img_test = cv2.imread(
        #     os.path.join(PATH_MASK, f"{imgname}.png"), cv2.IMREAD_GRAYSCALE
        # )
        # mask_img_test_dilated = Image_.dilate(
        #     mask_img_test, kernel=(7, 7), iterations=4
        # )
        # # cv2.imshow("orig", mask_img_test)
        # mask_img_test = mask_img_test_dilated - mask_img_test
        # rgb_img_for_ratio = logproc.apply_blur_with_mask(
        #     rgb_img, mask_img_test, kernel_size=(15, 15), iterations=10
        # )
        # cv2.imshow("xx", mask_img_test)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit()

        points = [lit_pt, shadow_pt]
        lit_shadow_pts = [
            log_img[points[-2][0], points[-2][1], :],
            log_img[points[-1][0], points[-1][1], :],
        ]
        ISD_vec = lit_shadow_pts[0] - lit_shadow_pts[1]

        #################################################################################################################

        # plane_eq_params = logproc.getPlane(ISD_vec, lit_shadow_pts[0])
        # logproc.generate_rotated_planes(plane_eq_params, log_img, imgname)
        # continue

        ###################################################################################################################

        # def normalize(pts_2d):
        #     x_vals, y_vals = pts_2d[:, 0], pts_2d[:, 1]
        #     min_val = np.array([np.min(x_vals), np.min(y_vals)])
        #     max_val = np.array([np.max(x_vals), np.max(y_vals)])
        #     return (pts_2d - min_val) / (max_val - min_val + 1e-8)

        # tfmat = logproc.estimateTransformationMat(ISD_vec)
        # projected_points_2d = logproc.transformPoints(log_img, tfmat)
        # # # projected_points_2d = normalize(projected_points_2d)
        # # gray_img_ip = logproc.estimateGrayscaleImagefromXY(
        # #     projected_points_2d, log_img.shape
        # # )
        # # projected_points_2d_new = logproc.plot2Dpts(
        # #     projected_points_2d, threshold=0.05, show_plot=False
        # # ).astype(np.float32)
        # # projected_points_2d_new = logproc.filterPointsbyNearestNeighbor(
        # #     projected_points_2d_new
        # # )
        # # gray_img_new_ip = logproc.estimateGrayscaleImagefromXY(
        # #     projected_points_2d_new, log_img.shape
        # # )
        # # xx = (log_img.shape[0], log_img.shape[1], 2)
        # # final_img_log = logproc.getfromlog(
        # #     projected_points_2d_new.reshape(xx), log_img, ratio
        # # )
        # # final_img_log = Image_.convertlogtolinear(final_img_log)
        # # cv2.imshow("final_img_log", final_img_log)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()
        # # continue
        # # projected_points_2d_new = normalize(projected_points_2d_new)

        # # logproc.apply_brightest_connected_components(
        # #     projected_points_2d_new.reshape(log_img.shape[0], log_img.shape[1], 2),
        # #     rgb_img,
        # #     intensity_threshold=50,
        # # )
        # # continue

        # tfmat = logproc.estimateTransformationMat(ISD_vec)
        # projected_points_2d_gt = logproc.transformPoints(ground_truth, tfmat)

        # # ISD_vec = np.array([0.5751, 0.5078, 0.6414], dtype=np.float32)
        # # tfmat = logproc.estimateTransformationMat(ISD_vec)
        # # projected_points_2d_gt = logproc.transformPoints(log_img, tfmat)
        # # # projected_points_2d_gt = normalize(projected_points_2d_gt)
        # # gray_img_gt = logproc.estimateGrayscaleImagefromXY(
        # #     projected_points_2d_gt, log_img.shape
        # # )

        # # projected_points_2d_gt_new = logproc.plot2Dpts(
        # #     projected_points_2d_gt, threshold=0.05, show_plot=False
        # # ).astype(np.float32)
        # # projected_points_2d_gt_new = logproc.filterPointsbyNearestNeighbor(
        # #     projected_points_2d_gt_new
        # # )
        # # gray_img_gt_new = logproc.estimateGrayscaleImagefromXY(
        # #     projected_points_2d_gt_new, log_img.shape
        # # )
        # # np.save(f"./plane_rotation/npy_gt/{imgname}.npy", projected_points_2d_gt_new)
        # # continue
        # # (
        # # projected_points_2d_new,
        # # projected_points_2d_gt_new,
        # # gray_img_ip_new,
        # # gray_img_gt_new,
        # # ) =
        # logproc.plot_two_sets_of_points(
        #     projected_points_2d, projected_points_2d_gt, imgname, rgb_img
        # )
        # continue

        # # cv2.imshow("xx", np.hstack([gray_img_ip_new, gray_img_gt_new]))
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()

        # # continue

        # # gray_img_ip_new = logproc.estimateGrayscaleImagefromXY(
        # #     projected_points_2d_new, log_img.shape
        # # )

        # projected_points_2d_new = projected_points_2d

        # # shape_val = (log_img.shape[0], log_img.shape[1], 2)
        # # xx = (log_img.shape[0], log_img.shape[1], 1)
        # # lastchannel = np.zeros(shape=xx)
        # # projected_points_2d_new = np.concatenate(
        # #     [projected_points_2d_new.reshape(shape_val), lastchannel], axis=2
        # # ).astype(np.float32)

        # # projected_points_2d_gt_new = np.concatenate(
        # #     [projected_points_2d_gt_new.reshape(shape_val), lastchannel], axis=2
        # # ).astype(np.float32)

        # # print(projected_points_2d_new.shape)
        # # print(projected_points_2d_new.dtype)
        # # print(projected_points_2d_gt_new.shape)
        # # print(projected_points_2d_gt_new.dtype)

        # # exit()
        # # tifffile.imwrite(
        # #     f"/home/satviktyagi/Desktop/desk/project/logchromaticity_guidance/ISTD/xy_map_data/train/input/{imgname}.tiff",
        # #     projected_points_2d_new,
        # # )
        # # tifffile.imwrite(
        # #     f"/home/satviktyagi/Desktop/desk/project/logchromaticity_guidance/ISTD/xy_map_data/train/gt/{imgname}.tiff",
        # #     projected_points_2d_gt_new,
        # # )
        # # cv2.imwrite(
        # #     f"loggray_GT/{imgname}.png",
        # #     gray_img_gt,
        # # )
        # # cv2.imshow("gray img ip", gray_img)
        # # cv2.imshow("gray img gt", gray_img_gt)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()
        # # continue
        ##################################################################################################################
        # projected_points_2d_new = normalize(projected_points_2d_new)
        # xy_map = projected_points_2d_new.reshape(log_img.shape[0], log_img.shape[1], 2)
        # print("xy_map min", np.min(xy_map))
        # print("xy_map max", np.max(xy_map))
        # # rgb_img_copy = rgb_img.copy()
        # # patch_sizes = [64, 128, 256]
        # # for j in range(3):
        # #     rgb_img_patches, _ = logproc.patchify(
        # #         rgb_img_copy, patch_size=patch_sizes[j], stride=32
        # #     )
        # #     xy_map_patches, _ = logproc.patchify(
        # #         xy_map, patch_size=patch_sizes[j], stride=32
        # #     )

        # #     final_img_patches = []
        # #     for xy_patch, rgb_patch in zip(xy_map_patches, rgb_img_patches):
        # #         final_img_patches.append(
        # #             logproc.set_pixels_based_on_xy_map(xy_patch, rgb_patch)
        # #         )
        # #     final_img = logproc.restitch(
        # #         final_img_patches,
        # #         (log_img.shape[0], log_img.shape[1]),
        # #         patch_size=patch_sizes[j],
        # #         stride=32,
        # #     )
        # #     rgb_img_copy = final_img.copy()
        # #     cv2.imshow("rgb_img_copy", rgb_img_copy)
        # #     cv2.waitKey(0)
        # #     cv2.destroyAllWindows()
        # # exit()

        # final_img = logproc.set_pixels_based_on_xy_map(
        #     xy_map, rgb_img, outlier_removal=False
        # )
        # # with_outlier = logproc.set_pixels_based_on_xy_map(
        # #     xy_map, rgb_img, outlier_removal=False
        # # )

        # final_img_init = final_img.copy()
        # # cv2.imshow("final_img_init", np.hstack([final_img_init, with_outlier]))
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()
        # # continue

        # final_img = logproc.removespots(
        #     final_img,
        #     mask_img,
        #     kernel_size=(3, 3),
        #     iterations=2,
        #     threshold_ratio=0.5,
        #     threshold_for_intensity_diff=1.25,
        # )

        # cv2.imshow("spot_removed", np.hstack([final_img_init, final_img]))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # continue

        # iterations = 3
        # applied_amount = [1, 1, 1]
        # kernel_size = [3, 3, 3]
        # for i in tqdm(range(iterations), desc="Applying ratio map"):
        #     ratio_map = logproc.calculate_ratio_value_map(
        #         rgb_img, final_img, kernel_size=kernel_size[i]
        #     )
        #     # if mask_img_test is not None:
        #     #     ratio_map[mask_img_test] = 1

        #     # ratio_map = 1 - ratio_map
        #     # ratio_map = np.where(ratio_map < 0, 0, ratio_map) * applied_amount[i]
        #     # ratio_map = ratio_map * applied_amount[i]

        #     ratio_map_expanded = ratio_map[
        #         :, :, np.newaxis
        #     ]  # Expand dims to (height, width, 1)

        #     final_img = final_img.astype(np.float32)

        #     final_img = np.multiply(final_img, ratio_map_expanded)
        #     # reprojected_rgbimg = reprojected_rgbimg + np.multiply(
        #     # reprojected_rgbimg, ratio_map_expanded
        #     # )

        # # reprojected_rgbimg = Image_.normalizeImage(reprojected_rgbimg)
        # # reprojected_rgbimg = reprojected_rgbimg.astype(np.uint8)

        # final_img = np.clip(final_img, 0, 255).astype(np.uint8)

        # # gray_as_brightness = logproc.set_gray_img_as_brightness(
        # # final_img, gray_img_new_gray, brightness_factor=1.2
        # # )

        # # cv2.imshow("gray_as_brightness", gray_as_brightness)
        # # cv2.imshow("final_img", final_img)
        # # cv2.imshow("final_img_init", final_img_init)
        # # cv2.imshow("gray_img_new", gray_img_new)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()

        # # gray_img_ip = cv2.cvtColor(gray_img_ip, cv2.COLOR_GRAY2BGR)
        # # gray_img_gt = cv2.cvtColor(gray_img_gt, cv2.COLOR_GRAY2BGR)
        # # gray_img_ip_new = cv2.cvtColor(gray_img_ip_new, cv2.COLOR_GRAY2BGR)
        # # gray_img_gt_new = cv2.cvtColor(gray_img_gt_new, cv2.COLOR_GRAY2BGR)
        # combined_img = np.hstack(
        #     [
        #         rgb_img,
        #         # gray_img_ip_new,
        #         # gray_img_gt_new,
        #         final_img_init,
        #         final_img,
        #     ]
        # )
        # # cv2.imwrite(
        # #     f"/home/satviktyagi/Desktop/desk/project/to_show/clustering_with_clip_and_min_nbors_correct_grayscaling/comparison/{imgname}.png",
        # #     combined_img,
        # # )
        # cv2.imshow(f"{imgname}", combined_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # # logproc.process_image(rgb_img, final_img)
        # # cv2.imwrite(f"./performance_comparison/train_A/{imgname}.png", combined_img)
        # # cv2.imwrite(f"./train_A/{imgname}.png", final_img)
        # continue

        ##############################################################################
        log_img = scale_to_255(log_img)
        points = [lit_pt, shadow_pt]
        lit_shadow_pts = [
            log_img[points[-2][0], points[-2][1], :],
            log_img[points[-1][0], points[-1][1], :],
        ]
        ISD_vec = lit_shadow_pts[0] - lit_shadow_pts[1]

        # moved_pt = qts = [lit_shadow_pts[0], moved_pt]
        a, b, c, d = logproc.getPlane(ISD_vec, lit_shadow_pts[1])

        projected_3D_pts = logproc.get3DpointsOnPlane(
            log_img.reshape(-1, 3), a, b, c, d
        )
        reprojected_logimg = projected_3D_pts.reshape(log_img.shape)

        plot_reproj_logimg = cv2.resize(reprojected_logimg, (100, 100))
        plot_log_img = cv2.resize(log_img, (100, 100))
        plot_rgb = cv2.resize(rgb_img, (100, 100))

        reprojected_logimg = reprojected_logimg.astype(np.uint8)
        reprojected_logimg_gray = np.mean(reprojected_logimg, axis=2)
        reprojected_logimg_gray = np.expand_dims(reprojected_logimg_gray, axis=2)
        reprojected_logimg_gray = (reprojected_logimg_gray).astype(np.uint8)

        # print(reprojected_logimg_gray.shape)
        # exit()

        reprojected_logimg_gray = cv2.cvtColor(
            reprojected_logimg_gray, cv2.COLOR_GRAY2BGR
        )
        gray_image = cv2.imread(
            f"/home/satviktyagi/Desktop/desk/project/datasets/des3_dataset/ISTD_des3/test/loggray/{imgname}.png",
            cv2.IMREAD_UNCHANGED,
        )
        rgb_final = getRefinedColorImg(
            reprojected_logimg, gray_image, imgname, clahe=False
        )
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([log_img, reprojected_logimg, gray_image, rgb_final])

        cv2.imshow("xx", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # cv2.imwrite(
        #     f"/home/satviktyagi/Desktop/desk/project/to_show/color_reprojected_images/{imgname}.png",
        #     combined,
        # )
        # continue

        logproc.generate3DHistogram(
            plot_reproj_logimg.reshape(-1, 3),
            plot_log_img.reshape(-1, 3),
            color_image=plot_rgb,
            save_name=f"reprojected_log_cloud.html",
            title="LOG PCL",
            vector=[lit_shadow_pts],
            planes=[[a, b, c, d]],
            center_point=lit_shadow_pts[0],
            plane_size=0,
        )
        # reprojected_logimg = np.exp(reprojected_logimg).astype(np.uint8)
        # cv2.imshow("reproj_rgb", reprojected_logimg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # reprojected_logimg = Image_.convertlogtolinear(reprojected_logimg)
        # reprojected_logimg = Image_.setconstantIntensity(
        #     reprojected_logimg, intensity=100
        # )
        # cv2.imshow("projlogimg", reprojected_logimg.astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # logproc.generate3DHistogram(
        #     log_img.reshape(-1, 3),
        #     color_image=rgb_img,
        #     save_name=f"logcloud.html",
        #     title="LOG PCL",
        #     vector=[lit_shadow_pts],
        #     planes=[[a, b, c, d]],
        #     center_point=lit_shadow_pts[0],
        # )
        # exit()


def scale_to_255(image):
    current_min = np.min(image, axis=(0, 1), keepdims=True)
    current_max = np.max(image, axis=(0, 1), keepdims=True)
    scaled_image = (image - current_min) / (current_max - current_min) * 255
    return np.clip(scaled_image, 0, 255).astype(np.uint8)


def apply_clahe(gray_image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_image = clahe.apply(gray_image)
    return clahe_image


def getRefinedColorImg(rgb_img, gray_image, imgname, clahe=False):
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))

    if clahe:
        gray_image = apply_clahe(gray_image)

    lab_planes[0] = gray_image
    lab = cv2.merge(lab_planes)
    rgb_img_final = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return rgb_img_final
    # cv2.imshow("final", np.hstack([rgb_img, gray_image, rgb_img_final]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # apply clahe
    # clahe_gray_image = apply_clahe(gray_image)


if __name__ == "__main__":
    main()
