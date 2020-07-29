import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import argparse


class LineMemory:
    def __init__(self):
        self.found_lanes = False
        self.last_coefficients = []


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
        [160, 720],  # Bottom Left
        [570, 460],  # Top Left
        [690, 460],  # Top Right
        [1090, 720]  # Bottom Right
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


def slidingWindow(img, nwindows=9, min_pixels=50, margin=100, visualize=True):
    window_height = img.shape[0] // nwindows

    histogram = createHistogram(img[img.shape[0]//2:, :])

    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    leftx_curr = leftx_base
    rightx_curr = rightx_base

    left_lane_idxs = []
    right_lane_idxs = []

    nonzero = img.nonzero()
    nonzeroy = nonzero[0]
    nonzerox = nonzero[1]

    out_img = np.dstack((img, img, img)) * 255

    for w in range(nwindows):
        topy = img.shape[0] - (w * window_height)
        bottomy = img.shape[0] - ((w + 1) * window_height)

        leftx_min = leftx_curr - margin
        leftx_max = leftx_curr + margin

        rightx_min = rightx_curr - margin
        rightx_max = rightx_curr + margin

        if visualize == True:
            cv2.rectangle(out_img, (leftx_min, topy),
                          (leftx_max, bottomy), (0, 255, 0), 4)
            cv2.rectangle(out_img, (rightx_min, topy),
                          (rightx_max, bottomy), (0, 255, 0), 4)

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

        if len(left_idxs) >= min_pixels:
            leftx_curr = np.int(np.mean(nonzerox[left_idxs]))

        if len(right_idxs) >= min_pixels:
            rightx_curr = np.int(np.mean(nonzerox[right_idxs]))

    left_lane_idxs = np.concatenate(left_lane_idxs)
    right_lane_idxs = np.concatenate(right_lane_idxs)

    left = (nonzerox[left_lane_idxs], nonzeroy[left_lane_idxs])
    right = (nonzerox[right_lane_idxs], nonzeroy[right_lane_idxs])

    return left, right, out_img


def fitPolynomial(size, leftx, lefty, rightx, righty):
    # Verical oriented
    (A1, B1, C1) = np.polyfit(lefty, leftx, 2)
    (A2, B2, C2) = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, size-1, size)

    left_fitx = A1 * ploty ** 2 + B1 * ploty + C1
    right_fitx = A2 * ploty ** 2 + B2 * ploty + C2

    return left_fitx, right_fitx


def main():
    parser = argparse.ArgumentParser("Do stuff")
    parser.add_argument("filename")

    args = parser.parse_args()
    filename = args.filename

    # Read a RGB image
    image = mpimg.imread(filename)
    image = cv2.resize(image, (1280, 720))
    # Convert image to HLS
    _, L, S = toHLS(image)

    # Apply Sobel operator on both L and S channels
    # And return the combined version of both binary images
    lbinary = to8BitSobel(absoluteSobel(L))
    sbinary = to8BitSobel(absoluteSobel(S))
    lbinary = valueThreshold(lbinary, (20, 100))
    sbinary = valueThreshold(sbinary, (20, 100))
    cbinary = cv2.bitwise_or(lbinary, sbinary)

    # Warp perspecive to see road from above
    warped = warpImage(cbinary)

    # Find lane lines points using sliding windows technique
    (leftx, lefty), (rightx, righty), out_img = slidingWindow(warped)

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Fit a 2nd degree polynomial
    left_line, right_line = fitPolynomial(img.shape[0], leftx,
                                          lefty, rightx, righty)

    plt.imshow(out_img)
    plt.plot(left_line, ploty, color="yellow")
    plt.plot(right_line, ploty, color="yellow")
    plt.show()


if __name__ == "__main__":
    main()
