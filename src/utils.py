import cv2
import numpy as np
import pickle

from glob import glob
from os import path


def toHLS(img):
    """Convert image to HLS channel
    and return separated channels"""
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    return H, L, S


def to8BitSobel(sobel):
    """Convert sobel output to 8 bit"""
    return np.uint8(255 * sobel / np.max(sobel))


def absoluteSobel(img, orient="x"):
    """Get absolute value of sobel operator"""
    if orient == "x":
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    return abs_sobel


def valueThreshold(img, thresh=(170, 256)):
    """Return a binary output for an
    arbitrary image threshold"""
    color_binary = np.zeros_like(img)
    color_binary[(img >= thresh[0]) & (img < thresh[1])] = 1
    return color_binary


def calibrateCamera(calibration_imgs_path):
    if path.isfile("matrix_dist.p"):
        file = open("matrix_dist.p", "rb")
        matrix_dist = pickle.load(file)
        file.close()

        mtx = matrix_dist["matrix"]
        dist = matrix_dist["dist"]

        return mtx, dist

    images = glob(calibration_imgs_path)

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


def undistortImage(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


def processImage(img):
    # Convert img to HLS
    img = cv2.GaussianBlur(img, (15, 15), 0)
    _, L, S = toHLS(img)
    # Apply Sobel operator on both L and S channels
    # And return the combined version of both binary images
    lbinary = to8BitSobel(absoluteSobel(L))
    sbinary = to8BitSobel(absoluteSobel(S))
    lbinary = valueThreshold(lbinary, (60, 100))
    sbinary = valueThreshold(sbinary, (60, 100))
    cbinary = cv2.bitwise_or(lbinary, sbinary)

    return cbinary


def warp(img, src, dst):
    """Change the perspective of an image
    to bird view"""

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img.shape[1::-1])
    return warped, Minv


def drawLane(img, warped, leftx, rightx, ploty, Minv):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([leftx, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([rightx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    newwarp = cv2.warpPerspective(
        color_warp, Minv, (warped.shape[1], warped.shape[0]))

    return cv2.addWeighted(img, 1, newwarp, 0.3, 0)
