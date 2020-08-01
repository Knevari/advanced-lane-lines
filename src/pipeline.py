import cv2
import numpy as np
import camera
import perspective
import utils
import visualization as vis

from lanefinder import LaneFinder


def processImage(img):
    # Convert img to HLS
    _, L, S = utils.toHLS(img)
    # Apply Sobel operator on both L and S channels
    # And return the combined version of both binary images
    L = cv2.GaussianBlur(L, (3, 3), 0)
    S = cv2.GaussianBlur(S, (3, 3), 0)

    l_sobelx = utils.absoluteSobel(L, "x", 3)
    s_sobelx = utils.absoluteSobel(S, "x", 3)

    lbinary = utils.to8BitSobel(l_sobelx)
    sbinary = utils.to8BitSobel(s_sobelx)
    lbinary = utils.valueThreshold(lbinary, (20, 100))
    sbinary = utils.valueThreshold(sbinary, (20, 100))

    cbinary = cv2.bitwise_or(lbinary, sbinary)

    return cbinary


class Pipeline:
    def __init__(self, height):
        self.mtx, self.dist = camera.calibrate("camera_cal/calibration*.jpg")
        self.ploty = np.linspace(0, height-1, height)
        self.lanefinder = LaneFinder(self.ploty, debug=True)

    def __call__(self, img):
        undistorted = camera.undistort(img, self.mtx, self.dist)
        thresh_img = processImage(undistorted)

        warped, Minv = perspective.warp(thresh_img)

        leftx_poly, rightx_poly = self.lanefinder.findLanes(warped)

        polygon = vis.drawLaneWithPolygon(
            warped, leftx_poly, rightx_poly, self.ploty)
        polygon = perspective.unwarp(polygon, Minv)

        combined = cv2.addWeighted(undistorted, 1.0, polygon, 0.3, 0)

        curvature = self.lanefinder.getCurvature()
        car_offset = self.lanefinder.getCarOffset()

        cv2.putText(combined, "Curvature Radius:" + "{:5.2f}km".format(curvature),
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    [255, 255, 255], 2, cv2.LINE_AA)

        cv2.putText(combined, "Distance from Center:" + '{:5.2f}cm'.format(car_offset * 100),
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    [255, 255, 255], 2, cv2.LINE_AA)

        return combined
