import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
import random

from utils.logchromaticity import processLogImage
from utils.utils import Image

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
PATH_LOG = (
    f"/home/satviktyagi/Desktop/desk/project/datasets/main_data/log_data/test/input"
)
PATH_RGB = (
    f"/home/satviktyagi/Desktop/desk/project/datasets/main_data/rgb_data/test/input"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Creating Log Chromaticity Image")
    parser.add_argument(
        "--result",
        default="results_generatedISD",
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
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="last processed image name",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    Image_ = Image()
    logproc = processLogImage()
    txt_file_path = os.path.join(args.result, "ISD_vec.txt")
    print(f"Saving ISD file at {txt_file_path}")

    if not os.path.exists(f"{args.result}"):
        os.makedirs(f"{args.result}")
        if args.savehistogram and not os.path.exists(f"{args.result}/histogram"):
            os.makedirs(f"{args.result}/histogram")

    imgnames = os.listdir(PATH_LOG)
    imgnames.sort()
    if args.resume:
        for i, name in enumerate(imgnames):
            name = name.split(".")[0]
            if name == args.resume:
                print(f"Resuming from img {imgnames[i+1]}")
                print(f"Processed {i+1} images in previous run")
                imgnames = imgnames[i + 1 :]
                break

    if not os.path.exists(txt_file_path):
        with open(txt_file_path, "w") as f:
            f.write(f"")

    for i in tqdm(range(len(imgnames)), desc="Processing images"):
        img_name = imgnames[i].split(".")[0]
        logimgpath = os.path.join(PATH_LOG, f"{img_name}.exr")
        rgbimgpath = os.path.join(PATH_RGB, f"{img_name}.png")

        log_img = cv2.imread(logimgpath, cv2.IMREAD_UNCHANGED)
        rgb_img = cv2.imread(rgbimgpath, cv2.IMREAD_UNCHANGED)

        while True:
            log_img_pts = log_img.reshape(-1, 3)

            #### MANUALLY SELECTING POINTS
            points = Image_.getPoints(rgb_img, num_points=2, title=img_name)
            lit_shadow_pts = [points[-2], points[-1]]  # lit, shadow
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
            gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

            Image_.showImage(gray_img)
            gray_rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
            gray_rgb_img = cv2.cvtColor(gray_rgb_img, cv2.COLOR_GRAY2BGR)
            compare = np.hstack([rgb_img, gray_rgb_img, gray_img])

            print("Save ISD for this image?")
            print("1: Yes")
            print("2: No")
            print("3: Yes and Exit")
            print("4: Exit")
            choice = input()
            if choice == "1" or choice == "3":
                print(f"Saving image {img_name}")
                with open(txt_file_path, "a") as f:
                    f.write(
                        f"{img_name} lit_pt, shadow_pt : {points[-2]} {points[-1]}\n"
                    )
                if choice == "3":
                    exit()
                break
            elif choice == "4":
                exit()
        Image_.saveImage(os.path.join(args.result, f"{img_name}_compare.png"), compare)
        continue


if __name__ == "__main__":
    main()
