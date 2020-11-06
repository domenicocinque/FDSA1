import numpy as np
from numpy import histogram as hist



#Add the Filtering folder, to import the gauss_module.py file, where gaussderiv is defined (needed for dxdy_hist)
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
filteringpath = os.path.join(parentdir, 'Filtering')
sys.path.insert(0,filteringpath)
import gauss_module



#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram

def normalized_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'
    len_bins = round(255/num_bins, 4)
    hists = np.zeros(num_bins)
    x = img_gray.reshape(img_gray.size) / len_bins
    x = np.array(x, dtype=int)
    bins = [len_bins*i for i in range(0,num_bins+1)]
    for el in x:
        hists[el] += 1

    hists = hists / sum(hists)
    return np.array(hists), np.array(bins)



#  Compute the *joint* histogram for each color channel in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
#
#  E.g. hists[0,9,5] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
#       - their B values fall in bin 5
def rgb_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'
    n, m = img_color_double.shape[0], img_color_double.shape[1]

    #Define a 3D histogram  with "num_bins^3" number of entries
    hists = np.zeros((num_bins, num_bins, num_bins))

    len_bins = round(255/num_bins, 4)
    x = img_color_double.reshape(1, n * m, 3)/len_bins
    x = np.array(x, dtype=int)
    # Loop for each pixel i in the image
    for i in range(n*m):
        i,j,k = x[0][i]
        hists[i,j,k] += 1

    hists = hists / np.sum(hists)
    hists = hists.reshape(hists.size)
    return hists



#  Compute the *joint* histogram for the R and G color channels in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^2
#
#  E.g. hists[0,9] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
def rg_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'

    n, m = img_color_double.shape[0], img_color_double.shape[1]

    # Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))

    len_bins = round(255 / num_bins, 4)
    x = img_color_double.reshape(1, n * m, 3) / len_bins
    x = np.array(x, dtype=int)
    for i in range(n * m):
        i, j, k = x[0][i]
        hists[i, j] += 1

    hists = hists / np.sum(hists)
    hists = hists.reshape(hists.size)
    return hists




#  Compute the *joint* histogram of Gaussian partial derivatives of the image in x and y direction
#  Set sigma to 3.0 and cap the range of derivative values is in the range [-6, 6]
#  The histogram should be normalized so that sum of all values equals 1
#
#  img_gray - input gray value image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#
#  Note: you may use the function gaussderiv from the Filtering exercise (gauss_module.py)
def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'
    Dx, Dy = gauss_module.gaussderiv(img_gray, 3.0)
    n = Dx.shape[0]
    Dx = Dx.reshape(1, n ** 2)[0]
    Dy = Dy.reshape(1, n ** 2)[0]
    Dxy = []
    for i in range(n):
        Dxy.append([Dx[i],Dy[i]])

    for elem in Dxy:
        for i in range(2):
            if elem[i] < -6:
                elem[i] = -6
            elif elem[i] > 6:
                elem[i] = 6

    #Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))

    len_bins = round(12 / num_bins, 4)
    bins = [-6]
    for i in range(1 , num_bins):
        bins.append(round(bins[i - 1] + len_bins, 4))
    x = np.array(np.array(Dxy) / len_bins)
    x = np.floor(x)
    x = np.array(x, dtype=int)

    for k in range(len(x)):
        i, j = x[k]
        hists[i, j] += 1

    hists = hists / np.sum(hists)
    hists = hists.reshape(hists.size)
    return hists



def is_grayvalue_hist(hist_name):
  if hist_name == 'grayvalue' or hist_name == 'dxdy':
    return True
  elif hist_name == 'rgb' or hist_name == 'rg':
    return False
  else:
    assert False, 'unknown histogram type'


def get_hist_by_name(img, num_bins_gray, hist_name):
  if hist_name == 'grayvalue':
    return normalized_hist(img, num_bins_gray)
  elif hist_name == 'rgb':
    return rgb_hist(img, num_bins_gray)
  elif hist_name == 'rg':
    return rg_hist(img, num_bins_gray)
  elif hist_name == 'dxdy':
    return dxdy_hist(img, num_bins_gray)
  else:
    assert False, 'unknown histogram: %s'%hist_name

