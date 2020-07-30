import numpy as np
from lines import searchPriorLines
from utils import calibrateCamera, processImage, undistortImage, warp, drawLane
from LaneMemory import LaneMemory


class Pipeline:
    def __init__(self, height):
        self.mtx, self.dist = calibrateCamera("camera_cal/calibration*.jpg")
        self.transform_coordinates = {
            "src": np.float32([
                [280, 700],
                [595, 460],
                [725, 460],
                [1125, 700]
            ]),
            "dst": np.float32([
                [250, 720],
                [250, 0],
                [1065, 0],
                [1065, 720]
            ])
        }
        self.ploty = np.linspace(0, height-1, height)
        self.memory = LaneMemory()

    def __call__(self, img):
        undistorted = undistortImage(img, self.mtx, self.dist)
        thresh_img = processImage(undistorted)

        src = self.transform_coordinates["src"]
        dst = self.transform_coordinates["dst"]
        warped, Minv = warp(thresh_img, src, dst)

        left_fit, right_fit, leftx_poly, rightx_poly, ploty = searchPriorLines(
            warped, self.memory.left, self.memory.right, self.ploty)

        lane_img = drawLane(undistorted, warped, leftx_poly,
                            rightx_poly, ploty, Minv)

        return lane_img
