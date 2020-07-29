import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import argparse


def toHLS(img):
    """Convert image to HLS channel
    and return separated channels"""
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    return H, L, S


def valueThreshold(img, thresh=(170, 256)):
    """Return a binary output for an 
    arbitrary image threshold"""
    color_binary = np.zeros_like(img)
    color_binary[(img >= thresh[0]) & (img < thresh[1])] = 1
    return color_binary


def absoluteSobel(img, orient="x"):
    """Get absolute value of sobel operator"""
    if orient == "x":
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    return abs_sobel


def to8BitSobel(sobel):
    """Convert sobel output to 8 bit"""
    return np.uint8(255 * sobel / np.max(sobel))


def createHistogram(img):
    """Create a histogram of the intensity of 
    the values in the image"""
    histogram = np.sum(img, axis=0)
    return histogram


def warpImage(img):
    """Change the perspective of an image
    to bird view"""
    offset = 40

    height = img.shape[0]
    width = img.shape[1]

    src = np.array([
        [165, 705],  # Bottom Left
        [595, 460],  # Top Left
        [735, 460],  # Top Right
        [1105, 705]  # Bottom Right
    ], dtype=np.float32)

    dst = np.array([
        [offset, height-offset],       # Bottom Left
        [offset, offset],              # Top Left
        [width-offset, offset],        # Top Right
        [width-offset, height-offset]  # Bottom Right
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img.shape[1::-1])
    return warped


def main():
    parser = argparse.ArgumentParser("Do stuff")
    parser.add_argument("filename")

    args = parser.parse_args()
    filename = args.filename

    # Read a RGB image
    image = mpimg.imread(filename)
    image = cv2.resize(image, (1280, 720))
    _, L, S = toHLS(image)

    lbinary = to8BitSobel(absoluteSobel(L))
    sbinary = to8BitSobel(absoluteSobel(S))
    lbinary = valueThreshold(lbinary, (20, 100))
    sbinary = valueThreshold(sbinary, (20, 100))
    cbinary = cv2.bitwise_or(lbinary, sbinary)

    warped = warpImage(cbinary)
    histogram = createHistogram(warped[warped.shape[0]//2:, :])

    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    box_margin = 100
    min_pixels = 50
    nwindows = 9
    window_height = cbinary.shape[0] // nwindows

    leftx_curr = leftx_base
    rightx_curr = rightx_base

    left_lane_idxs = []
    right_lane_idxs = []

    nonzero = warped.nonzero()
    nonzeroy = nonzero[0]
    nonzerox = nonzero[1]

    for w in range(nwindows):
        topy = warped.shape[0] - (w * window_height)
        bottomy = warped.shape[0] - ((w + 1) * window_height)

        leftx_min = leftx_curr - box_margin
        leftx_max = leftx_curr + box_margin

        rightx_min = rightx_curr - box_margin
        rightx_max = rightx_curr + box_margin

        cv2.rectangle(warped, (leftx_min, topy), (leftx_max, bottomy), (0, 255, 0), 4)
        cv2.rectangle(warped, (rightx_min, topy), (rightx_max, bottomy), (0, 255, 0), 4)

        left_idxs = (nonzeroy >= bottomy) & \
                    (nonzeroy < topy) & \
                    (nonzerox >= leftx_min) & \
                    (nonzerox > leftx_max)

        right_idxs = (nonzeroy >= bottomy) & \
                     (nonzeroy < topy) & \
                     (nonzerox >= rightx_min) & \
                     (nonzerox > rightx_max)

        left_lane_idxs.append(left_idxs)
        right_lane_idxs.append(right_idxs)

        if len(left_idxs) >= min_pixels:
            leftx_curr = np.int(np.mean(nonzerox[left_idxs]))

        if len(right_idxs) >= min_pixels:
            rightx_curr = np.int(np.mean(nonzerox[right_idxs]))

    plt.imshow(warped)
    plt.show()


if __name__ == "__main__":
    main()
