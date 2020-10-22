import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2

"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""


def gauss(sigma):
    sigma = int(sigma)
    Gx = np.array([])
    x = np.array([])
    for i in range(-3 * sigma, 3 * sigma + 1):
        Gx = np.append(Gx, (1 / math.sqrt(2 * math.pi * sigma ** 2)) * math.exp(-i ** 2 / (2 * sigma ** 2)))
        x = np.append(x, i)
    return Gx, x


"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""


def gaussianfilter(img, sigma):
    smooth_img = []  # TODO: Why can't we start directly with a np array?
    for i in range(len(img)):
        smooth_img.append(np.convolve(img[i], gauss(sigma)[0], mode='valid'))
    smooth_img = np.transpose(np.array(smooth_img))
    smooth_img2 = []
    for i in range(len(smooth_img)):
        smooth_img2.append(np.convolve(smooth_img[i], gauss(sigma)[0], mode='valid'))
    return list(np.transpose(np.array(smooth_img2)))


"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""


def gaussdx(sigma):
    sigma = int(sigma)
    Dx = np.array([])
    x = np.array([])
    for i in range(-3 * sigma, 3 * sigma + 1):
        Dx = np.append(Dx, (- i / math.sqrt(2 * math.pi * sigma ** 2)) * math.exp(-i ** 2 / (2 * sigma ** 2)))  # TODO: Why sigma^3 on the text?
        x = np.append(x, i)
    return Dx, x


def gaussderiv(img, sigma):
    # ...

    return imgDx, imgDy
