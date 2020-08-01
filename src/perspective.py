import cv2
import numpy as np

src = np.float32([
    [280, 700],
    [595, 460],
    [725, 460],
    [1125, 700]
])

dst = np.float32([
    [250, 720],
    [250, 0],
    [1065, 0],
    [1065, 720]
])


def warp(img):
    """Change the perspective of an image
    to bird view"""

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(
        img, M, img.shape[1::-1], flags=cv2.INTER_NEAREST)
    return warped, Minv


def unwarp(warped, Minv):
    return cv2.warpPerspective(warped, Minv, (warped.shape[1], warped.shape[0]))
