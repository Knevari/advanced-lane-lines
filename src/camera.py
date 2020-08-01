import cv2
import numpy as np
import pickle

from glob import glob
from os import path


def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


def calibrate(calibration_path):
    if path.isfile("matrix_dist.p"):
        file = open("matrix_dist.p", "rb")
        matrix_dist = pickle.load(file)
        file.close()

        mtx = matrix_dist["matrix"]
        dist = matrix_dist["dist"]

        return mtx, dist

    images = glob(calibration_path)

    img_points = []
    obj_points = []

    objp = np.zeros((54, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    for fname in images:
        image = cv2.imread(fname)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret == True:
            img_points.append(corners)
            obj_points.append(objp)

    test_img = cv2.imread(images[0])
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    ret, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points,
                                               test_img.shape[::-1], None, None)

    file = open("matrix_dist.p", "wb")
    pickle.dump({"matrix": mtx, "dist": dist}, file)
    file.close()

    return mtx, dist
