import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm

from logchromaticity import processLogImage
from utils import Image

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
PATH_RGB = f"/home/satviktyagi/Desktop/desk/project/datasets/des3_dataset/ISTD_des3/test/test_A/"


def parse_args():
    parser = argparse.ArgumentParser(description="Creating Log Chromaticity Image")
    parser.add_argument(
        "--img",
        default=None,
        type=str,
        help="name (if exist in same dir) or path of image to process",
    )
    parser.add_argument(
        "--imgs",
        default=None,
        type=str,
        help="image folder",
    )
    parser.add_argument(
        "--result",
        default=None,
        type=str,
        help="image folder",
    )
    parser.add_argument(
        "--crop",
        default=None,
        type=str,
        help="flag to set crop the image to true or false",
    )
    parser.add_argument(
        "--savehistogram",
        default=None,
        type=str,
        help="flag to set save histogram as true or false",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    Image_ = Image()
    logproc = processLogImage()

    if not os.path.exists(f"{args.result}"):
        os.makedirs(f"{args.result}")
        if args.savehistogram and not os.path.exists(f"{args.result}/histogram"):
            os.makedirs(f"{args.result}/histogram")

    if args.img is not None:
        if "/" in args.img:
            img_name = (os.path.basename(args.img)).split(".")[0]
        elif "." in args.img:
            img_name = (args.img).split(".")[0]

        path = args.img
        log_img = Image_.readImage(path)
        rgb_img = Image_.readImage(os.path.join(PATH_RGB, f"{img_name}.png"))

        if args.crop is not None:
            x1, y1, x2, y2 = Image_.getCrop(rgb_img)
            log_img = log_img[min(y1, y2) : max(y1, y2), min(x1, x2) : max(x1, x2)]
            rgb_img = rgb_img[min(y1, y2) : max(y1, y2), min(x1, x2) : max(x1, x2)]

        logrgbimgs = [(log_img, rgb_img, img_name)]

    elif args.imgs is not None:
        imgnames = os.listdir(args.imgs)
        logrgbimgs = []
        for im in imgnames:
            imname = im.split(".")[0]
            logimgpath = os.path.join(args.imgs, im)
            rgbimgpath = os.path.join(PATH_RGB, f"{imname}.png")

            log_img = Image_.readImage(logimgpath)
            rgb_img = Image_.readImage(rgbimgpath)

            logrgbimgs.append((log_img, rgb_img, imname))

    for log_img, rgb_img, img_name in tqdm(logrgbimgs, desc="Processing images"):
        #### MANUALLY SELECTING POINTS
        # points = Image_.getPoints(rgb_img, num_points=2)
        # lit_shadow_pts = [points[-2], points[-1]]  # lit, shadow
        # ISD_vec = logproc.estimateISDwithNeigbors(
        #     log_img, lit_shadow_pts, kernel=(5, 5)
        # )
        # lit_shadow_pts = [
        #     log_img[points[-2][0], points[-2][1], :],
        #     log_img[points[-1][0], points[-1][1], :],
        # ]

        #### PLANE FITTING
        # log_img_pts = log_img.reshape(-1, 3)
        # sufficient_inliers = log_img_pts.shape[0]
        # best_fit_plane, best_inliers_mask = logproc.fitPlane(
        #     log_img_pts, sufficient_inliers, max_iterations=1000, threshold=0.01
        # )
        # best_projection_plane = logproc.getNormaltoPlane(best_fit_plane)
        # best_projection_plane_normal = best_projection_plane[0:3]

        # tfmat = logproc.estimateTransformationMat(best_projection_plane_normal)
        # projected_points_2d = logproc.transformPoints(log_img, tfmat)

        # intensity_projected_pts_2d = np.mean(projected_points_2d, axis=1)

        # inliers_indices = np.where(best_inliers_mask)[0]

        # min_intensity_in_filtered = np.argmin(
        #     intensity_projected_pts_2d[best_inliers_mask]
        # )
        # min_intensity_idx = inliers_indices[min_intensity_in_filtered]

        # max_intensity_in_filtered = np.argmax(
        #     intensity_projected_pts_2d[best_inliers_mask]
        # )
        # max_intensity_idx = inliers_indices[max_intensity_in_filtered]

        # litpt = log_img_pts[max_intensity_idx, :]
        # shadowpt = log_img_pts[min_intensity_idx, :]
        # lit_shadow_pts = [litpt, shadowpt]
        # ISD_vec = shadowpt - litpt

        ### LINE FITTING
        log_img_pts = log_img.reshape(-1, 3)
        lit_shadow_pts, best_inliers_mask = logproc.fitLine(
            log_img_pts, max_iterations=5000, threshold=1
        )
        ISD_vec = lit_shadow_pts[0] - lit_shadow_pts[1]

        tfmat = logproc.estimateTransformationMat(ISD_vec)
        projected_points_2d = logproc.transformPoints(log_img, tfmat)

        point_color_dict = {}
        min_x, max_x = np.min(projected_points_2d[:, 0]), np.max(
            projected_points_2d[:, 0]
        )
        min_y, max_y = np.min(projected_points_2d[:, 1]), np.max(
            projected_points_2d[:, 1]
        )

        for point_2d, point_3d in tqdm(
            zip(projected_points_2d, rgb_img.reshape(-1, 3)),
            desc="Retreiving Original Colors",
        ):
            x, y = point_2d[0], point_2d[1]
            x_normalized = int(((x - min_x) / (max_x - min_x)) * 255)
            y_normalized = int(((y - min_y) / (max_y - min_y)) * 255)
            norm_point_2d = (x_normalized, y_normalized)
            tuple_pt_3d = tuple(point_3d)
            if norm_point_2d not in point_color_dict:
                point_color_dict[norm_point_2d] = set([tuple_pt_3d])
            elif tuple_pt_3d not in point_color_dict[norm_point_2d]:
                point_color_dict[norm_point_2d].add(tuple_pt_3d)

        gray_img = logproc.estimateGrayscaleImagefromXY(
            projected_points_2d, log_img.shape
        )
        Image_.saveImage(os.path.join(args.result, f"{img_name}_gray.png"), gray_img)
        Image_.showImage(gray_img)
        projected_points_RGB = logproc.reprojectXYtoOriginalColor(
            projected_points_2d, point_color_dict, intensity=None
        )
        color_img = projected_points_RGB.reshape(log_img.shape)
        Image_.saveImage(
            os.path.join(args.result, f"{img_name}_logchrom.png"), color_img
        )
        Image_.showImage(color_img, destroy=True)
        Image_.saveImage(os.path.join(args.result, f"{img_name}_original.png"), rgb_img)

        if args.savehistogram:
            logproc.generate3DHistogram(
                log_img,
                color_image=rgb_img,
                save_name=f"{args.result}/histogram/{img_name}_log.html",
                title="LOG PCL",
                vector=[lit_shadow_pts],
                # plane=[best_fit_plane, best_projection_plane],
            )
            logproc.generate3DHistogram(
                rgb_img,
                save_name=f"{args.result}/histogram/{img_name}_rgb.html",
                title="RGB PCL",
            )


if __name__ == "__main__":
    main()