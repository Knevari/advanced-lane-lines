import cv2
import numpy as np
import perspective


def drawLaneWithPolygon(warped, leftx, rightx, y_values):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([leftx, y_values]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([rightx, y_values])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    return color_warp
