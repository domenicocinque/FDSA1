import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import histogram_module
import dist_module


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain 
#       handles to distance and histogram functions, and to find out whether histogram function 
#       expects grayvalue or color image

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):
    hist_isgray = histogram_module.is_grayvalue_hist(hist_type)

    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)

    D = np.zeros((len(model_images), len(query_images)))
    best_match = []
    # DNAMES (Modificare compute histograms)

    for i in range(len(model_hists)):
        for j in range(len(query_hists)):
            D[i, j] = dist_module.get_dist_by_name(query_hists[j], model_hists[i], dist_type)

    D = np.transpose(D)
    for i in range(len(D)):
        minim = np.min(D[i])
        best_match.append(np.where(D[i] == minim)[0][0])

    return best_match, D


def compute_histograms(image_list, hist_type, hist_isgray, num_bins):
    image_hist = []

    for i in image_list:
        image_hist.append(histogram_module.get_hist_by_name(np.array(Image.open(i)).astype('double'), num_bins, hist_type))

    return image_hist


# For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# Note: use the previously implemented function 'find_best_match'
# Note: use subplot command to show all the images in the same Python figure, one row per query image

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):
    plt.figure()

    num_nearest = 5  # show the top-5 neighbors
    D = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)[1]
    OD = []

    for row in D:
        DS = np.sort(row)
        OD.append(DS[:num_nearest])

    five_best = []
    for row in OD:
        sl = []
        for el in row:
            index = np.where(D==el)
            sl.append(model_images[int(index[1])])
        five_best.append(sl)

    for i_query in range(len(query_images)):
        fig = plt.figure(i_query)
        plt.subplot(1,6,1)
        plt.imshow(np.array(Image.open(query_images[i_query])), vmin = 0, vmax = 255)
        for n in range(2, 7):
            plt.subplot(1,6,n)
            plt.imshow(np.array(Image.open(five_best[i_query][n-2])), vmin = 0, vmax = 255)
        #fig.add_subplot(i_query, 6, n)
    plt.show()

