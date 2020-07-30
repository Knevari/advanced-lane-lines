import numpy as np


def detectLines(img, left_lane, right_lane, ploty):
    histogram = np.sum(img[img.shape[0]//2:, :], axis=0)

    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    minpix = 50
    window_height = img.shape[0] // nwindows
    margin = 50

    leftx_curr = leftx_base
    rightx_curr = rightx_base

    nonzero = img.nonzero()
    nonzeroy = nonzero[0]
    nonzerox = nonzero[1]

    left_lane_idxs = []
    right_lane_idxs = []

    for w in range(nwindows):
        topy = img.shape[0] - (w * window_height)
        bottomy = img.shape[0] - ((w + 1) * window_height)

        leftx_min = leftx_curr - margin
        leftx_max = leftx_curr + margin

        rightx_min = rightx_curr - margin
        rightx_max = rightx_curr + margin

        left_idxs = ((nonzeroy >= bottomy) &
                     (nonzeroy < topy) &
                     (nonzerox >= leftx_min) &
                     (nonzerox < leftx_max)).nonzero()[0]

        right_idxs = ((nonzeroy >= bottomy) &
                      (nonzeroy < topy) &
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
        left_lane.detected = True

    if len(right_lane_idxs) > 0:
        right_lane.detected = True

    leftx = nonzerox[left_lane_idxs]
    lefty = nonzeroy[left_lane_idxs]

    rightx = nonzerox[right_lane_idxs]
    righty = nonzeroy[right_lane_idxs]

    (A1, B1, C1) = np.polyfit(lefty, leftx, 2)
    (A2, B2, C2) = np.polyfit(righty, rightx, 2)

    leftx_poly = A1 * ploty ** 2 + B1 * ploty + C1
    rightx_poly = A2 * ploty ** 2 + B2 * ploty + C2

    left_fit = (A1, B1, C1)
    right_fit = (A2, B2, C2)

    left_lane.best_fit = left_fit
    right_lane.best_fit = right_fit

    return left_fit, right_fit, leftx_poly, rightx_poly, ploty


def searchPriorLines(img, left_lane, right_lane, ploty):
    margin = 100

    if len(left_lane.best_fit) == 0 or len(right_lane.best_fit) == 0:
        return detectLines(img, left_lane, right_lane, ploty)

    (A1, B1, C1) = left_lane.best_fit
    (A2, B2, C2) = right_lane.best_fit

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_poly = A1 * nonzeroy ** 2 + B1 * nonzeroy + C1
    left_poly_min = left_poly - margin
    left_poly_max = left_poly + margin

    right_poly = A2 * nonzeroy ** 2 + B2 * nonzeroy + C2
    right_poly_min = right_poly - margin
    right_poly_max = right_poly + margin

    left_lane_idxs = (nonzerox > left_poly_min) & \
                     (nonzerox < left_poly_max)

    right_lane_idxs = (nonzerox > right_poly_min) & \
                      (nonzerox < right_poly_max)

    if (len(left_lane_idxs) == 0) or (len(right_lane_idxs) == 0):
        return detectLines(img, left_lane, right_lane, ploty)

    leftx = nonzerox[left_lane_idxs]
    lefty = nonzeroy[left_lane_idxs]

    rightx = nonzerox[right_lane_idxs]
    righty = nonzeroy[right_lane_idxs]

    (A1, B1, C1) = np.polyfit(lefty, leftx, 2)
    (A2, B2, C2) = np.polyfit(righty, rightx, 2)

    leftx_poly = A1 * ploty ** 2 + B1 * ploty + C1
    rightx_poly = A2 * ploty ** 2 + B2 * ploty + C2

    left_fit = (A1, B1, C1)
    right_fit = (A2, B2, C2)

    left_lane.best_fit = left_fit
    right_lane.best_fit = right_fit

    return left_fit, right_fit, leftx_poly, rightx_poly, ploty
