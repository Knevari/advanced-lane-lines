import cv2
import numpy as np


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


def absoluteSobel(img, orient="x", ksize=3):
    """Get absolute value of sobel operator"""
    if orient == "x":
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_sobel = np.absolute(sobel)
    return abs_sobel


def valueThreshold(img, thresh=(170, 256)):
    """Return a binary output for an
    arbitrary image threshold"""
    color_binary = np.zeros_like(img)
    color_binary[(img >= thresh[0]) & (img < thresh[1])] = 1
    return color_binary
