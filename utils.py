import cv2
import numpy as np
import os

points = []
drawing = False
ix, iy = -1, -1
rect = None


class Image:
    def readImage(self, path, read=cv2.IMREAD_UNCHANGED):
        img = cv2.imread(path, read)
        if img is None:
            raise Exception(f"Image at path {path} could not be read.")
        return img

    def showImage(self, img, windowname="img", destroy=True):
        if not isinstance(img, np.ndarray):
            raise Exception(f"Expected image of type np.ndarray, but got {type(img)}")
        if img.dtype != np.uint8:
            raise Exception(f"Expected image with dtype np.uint8, but got {type(img)}")
        cv2.imshow(windowname, img)
        cv2.waitKey(0)
        if destroy:
            cv2.destroyAllWindows()

    def converttouint8(self, image):
        return image.astype(np.uint8)

    def normalizeImage(self, image, minval=0, maxval=255, norm_type=cv2.NORM_MINMAX):
        return cv2.normalize(image, None, minval, maxval, norm_type)

    def saveImage(self, path, img):
        cv2.imwrite(path, img)

    def getCrop(self, rgbimg):
        def draw_rectangle(event, x, y, flags, param):
            global ix, iy, drawing, rect

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = x, y

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    img_copy = rgbimg.copy()
                    cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
                    cv2.imshow("image", img_copy)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                rect = (ix, iy, x, y)
                cv2.rectangle(rgbimg, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow("image", rgbimg)

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", draw_rectangle)

        while True:
            cv2.imshow("image", rgbimg)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c") and rect is not None:
                x1, y1, x2, y2 = rect
                cv2.destroyAllWindows()
                break
            if key == ord("q"):
                break
        return x1, y1, x2, y2

    def getPoints(self, rgb_img, num_points=2):
        def select_points(event, x, y, flags, param):
            global points
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((y, x))  # row, col
                print(f"Point selected: {x}, {y}")
                cv2.circle(rgb_img, (x, y), 3, (0, 255, 0), -1)
                cv2.imshow("img", rgb_img)

        cv2.namedWindow("img")
        cv2.setMouseCallback("img", select_points)
        while True:
            cv2.imshow("img", rgb_img)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("c") and len(points) == num_points:
                cv2.destroyAllWindows()
                break
            if key == ord("q"):
                break
        return points
