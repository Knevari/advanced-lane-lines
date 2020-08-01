import cv2
import numpy as np
import numpy.polynomial.polynomial as poly
from lanememory import LaneMemory

my = 30 / 720
mx = 3.7 / 780


class LaneFinder:
    def __init__(self, y_values, debug=False):
        self.memory = LaneMemory()
        self.y_values = y_values
        self.debug = debug

    def findLanes(self, image):
        if len(self.memory.left.best_fit) == 0:
            return self._applySlidingWindows(image)
        if len(self.memory.right.best_fit) == 0:
            return self._applySlidingWindows(image)

        return self._searchFromPrior(image)

    def getCurvature(self):
        y_eval = np.max(self.y_values)
        left_x = self.memory.left.xpoly
        right_x = self.memory.right.xpoly

        left_fit = np.polyfit(self.y_values * my, left_x * mx, 2)
        right_fit = np.polyfit(self.y_values * my, right_x * mx, 2)

        left_curv = ((1 + (2 * left_fit[0] * y_eval * my + left_fit[1])
                      ** 2) ** 1.5) / np.absolute(2 * left_fit[0])

        right_curv = ((1 + (2 * right_fit[0] * y_eval * my + right_fit[1])
                       ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

        return (0.5 / (left_curv / 1000 + right_curv / 1000))

    def getCarOffset(self):
        y_eval = np.max(self.y_values)

        left_fit = self.memory.left.best_fit
        right_fit = self.memory.right.best_fit

        left_fit = np.flip(left_fit, 0)
        right_fit = np.flip(right_fit, 0)
        leftl = poly.polyval(y_eval, left_fit)
        rightl = poly.polyval(y_eval, right_fit)

        center_road = leftl+((rightl-leftl)/2)
        center_car = 660
        caroff = (center_car-center_road) * mx

        return caroff

    def _applySlidingWindows(self, image):
        histogram = np.sum(image[image.shape[0]//2:, :], axis=0)

        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        minpix = 50
        window_height = image.shape[0] // nwindows
        margin = 50

        leftx_curr = leftx_base
        rightx_curr = rightx_base

        nonzero = image.nonzero()
        nonzeroy = nonzero[0]
        nonzerox = nonzero[1]

        left_lane_idxs = []
        right_lane_idxs = []

        for w in range(nwindows):
            y_top = image.shape[0] - (w * window_height)
            y_bottom = image.shape[0] - ((w + 1) * window_height)

            leftx_min = leftx_curr - margin
            leftx_max = leftx_curr + margin

            rightx_min = rightx_curr - margin
            rightx_max = rightx_curr + margin

            left_idxs = ((nonzeroy >= y_bottom) &
                         (nonzeroy < y_top) &
                         (nonzerox >= leftx_min) &
                         (nonzerox < leftx_max)).nonzero()[0]

            right_idxs = ((nonzeroy >= y_bottom) &
                          (nonzeroy < y_top) &
                          (nonzerox >= rightx_min) &
                          (nonzerox < rightx_max)).nonzero()[0]

            left_lane_idxs.append(left_idxs)
            right_lane_idxs.append(right_idxs)

            if len(left_idxs) >= minpix:
                leftx_curr = np.int(np.mean(nonzerox[left_idxs]))

            if len(right_idxs) >= minpix:
                rightx_curr = np.int(np.mean(nonzerox[right_idxs]))

        left_lane_idxs = np.concatenate(left_lane_idxs)
        right_lane_idxs = np.concatenate(right_lane_idxs)

        if len(left_lane_idxs) > 0:
            self.memory.left.detected = True
            self.memory.left.best_fit = None

        if len(right_lane_idxs) > 0:
            self.memory.right.detected = True
            self.memory.right.best_fit = None

        leftx = nonzerox[left_lane_idxs]
        lefty = nonzeroy[left_lane_idxs]

        rightx = nonzerox[right_lane_idxs]
        righty = nonzeroy[right_lane_idxs]

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        leftx_poly = left_fit[0] * self.y_values ** 2 + \
            left_fit[1] * self.y_values + \
            left_fit[2]

        rightx_poly = right_fit[0] * self.y_values ** 2 + \
            right_fit[1] * self.y_values + \
            right_fit[2]

        self.memory.left.setLastFit(left_fit)
        self.memory.left.setPoly(leftx_poly)

        self.memory.right.setLastFit(right_fit)
        self.memory.right.setPoly(rightx_poly)

        return leftx_poly, rightx_poly

    def _searchFromPrior(self, image):
        margin = 100

        if len(self.memory.left.best_fit) == 0:
            return self._applySlidingWindows(image)

        if len(self.memory.right.best_fit) == 0:
            return self._applySlidingWindows(image)

        left_fit = self.memory.left.best_fit
        right_fit = self.memory.right.best_fit

        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_poly = left_fit[0] * nonzeroy ** 2 + \
            left_fit[1] * nonzeroy + \
            left_fit[2]
        left_poly_min = left_poly - margin
        left_poly_max = left_poly + margin

        right_poly = right_fit[0] * nonzeroy ** 2 + \
            right_fit[1] * nonzeroy + \
            right_fit[2]
        right_poly_min = right_poly - margin
        right_poly_max = right_poly + margin

        left_lane_idxs = (nonzerox > left_poly_min) & \
            (nonzerox < left_poly_max)

        right_lane_idxs = (nonzerox > right_poly_min) & \
            (nonzerox < right_poly_max)

        leftx = nonzerox[left_lane_idxs]
        lefty = nonzeroy[left_lane_idxs]

        rightx = nonzerox[right_lane_idxs]
        righty = nonzeroy[right_lane_idxs]

        if len(rightx) == 0 or \
           len(righty) == 0 or \
           len(leftx) == 0 or \
           len(lefty) == 0:
            return self._applySlidingWindows(image)

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        leftx_poly = left_fit[0] * self.y_values ** 2 + \
            left_fit[1] * self.y_values + \
            left_fit[2]

        rightx_poly = right_fit[0] * self.y_values ** 2 + \
            right_fit[1] * self.y_values + \
            right_fit[2]

        self.memory.left.setLastFit(left_fit)
        self.memory.left.setPoly(leftx_poly)

        self.memory.right.setLastFit(right_fit)
        self.memory.right.setPoly(rightx_poly)

        if self.debug:
            out_img = np.dstack((image, image, image))*255
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_idxs],
                    nonzerox[left_lane_idxs]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_idxs],
                    nonzerox[right_lane_idxs]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array(
                [np.transpose(np.vstack([leftx_poly-margin, self.y_values]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([leftx_poly+margin,
                                                                            self.y_values])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array(
                [np.transpose(np.vstack([rightx_poly-margin, self.y_values]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([rightx_poly+margin,
                                                                             self.y_values])))])
            right_line_pts = np.hstack(
                (right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

            cv2.imshow("Lane Lines", result)

        return leftx_poly, rightx_poly
# Caso não consiga encontrar linhas nem no detect ou no prior, então retorna estado passado
# Salvar estado passado no LaneMemory
