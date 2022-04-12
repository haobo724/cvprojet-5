'''
@author: Prathmesh R Madhu.
For educational purposes only
'''
# -*- coding: utf-8 -*-
from __future__ import division

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy as np

def generate_segments(im_orig, scale, sigma, min_size):
    """
    Task 1: Segment smallest regions by the algorithm of Felzenswalb.
    1.1. Generate the initial image mask using felzenszwalb algorithm
    1.2. Merge the image mask to the image as a 4th channel
    """
    ### YOUR CODE HERE ###
    segments_fz = skimage.segmentation.felzenszwalb( skimage.util.img_as_float(im_orig), scale=scale, sigma=sigma, min_size=min_size)
    im_orig=np.dstack((im_orig,segments_fz))
    print(im_orig.shape)

    return im_orig

def sim_colour(r1, r2):
    """
    2.1. calculate the sum of histogram intersection of colour
    """
    ### YOUR CODE HERE ###

    return sum([min(a, b) for a, b in zip(r1["colorHist"], r2["colorHist"])])


def sim_texture(r1, r2):
    """
    2.2. calculate the sum of histogram intersection of texture
    """
    ### YOUR CODE HERE ###

    return sum([min(a, b) for a, b in zip(r1["textureHist"], r2["textureHist"])])


def sim_size(r1, r2, imsize):
    """
    2.3. calculate the size similarity over the image
    """
    ### YOUR CODE HERE ###

    return 1 - (r1["size"] + r2["size"]) / imsize


def sim_fill(r1, r2, imsize):
    """
    2.4. calculate the fill similarity over the image
    """
    ### YOUR CODE HERE ###

    max_x = max(r1["max_x"], r2["max_x"])
    min_x = min(r1["min_x"], r2["min_x"])
    max_y = max(r1["max_y"], r2["max_y"])
    min_y = min(r1["min_y"], r2["min_y"])
    bbSize = (max_x - min_x) * (max_y - min_y)
    return 1 - (bbSize - r1["size"] - r2["size"]) / imsize

def calc_sim(r1, r2, imsize):
    return (sim_colour(r1, r2) + sim_texture(r1, r2)
            + sim_size(r1, r2, imsize) + sim_fill(r1, r2, imsize))

def calc_colour_hist(img):
    """
    Task 2.5.1
    calculate colour histogram for each region
    the size of output histogram will be BINS * COLOUR_CHANNELS(3)
    number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
    extract HSV
    """
    BINS = 25
    hist = np.array([])
    ### YOUR CODE HERE ###
    for c in range(3):
        # h, _ = np.histogram(img[:, c], bins=BINS)
        # hist=np.append(hist,h)
        hist = np.concatenate(
            [hist] + [np.histogram(img[:, c], BINS, (0.0, 1.0))[0]])

    return hist/ len(img)

def calc_texture_gradient(img):
    """
    Task 2.5.2
    calculate texture gradient for entire image
    The original SelectiveSearch algorithm proposed Gaussian derivative
    for 8 orientations, but we will use LBP instead.
    output will be [height(*)][width(*)]
    Useful function: Refer to skimage.feature.local_binary_pattern documentation
    """
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    ### YOUR CODE HERE ###
    for i in range(3):
        ret[:, :, i] = skimage.feature.local_binary_pattern(img[:, :, i], P=8, R=1)
    return ret


def calc_texture_hist(img):
    """
    Task 2.5.3
    calculate texture histogram for each region
    calculate the histogram of gradient for each colours
    the size of output histogram will be
        BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    Do not forget to L1 Normalize the histogram
    """
    BINS = 10
    hist = np.array([])
    ### YOUR CODE HERE ###
    for c in range(3):
        # h, _ = np.histogram(img[:, c], bins=BINS)
        # hist=np.append(hist,h)
        hist = np.concatenate(
            [hist] + [np.histogram(img[:, c], BINS, (0.0, 1.0))[0]])
    return hist/ len(img)

def extract_regions(img):
    '''
    Task 2.5: Generate regions denoted as datastructure R
    - Convert image to hsv color map
    - Count pixel positions
    - Calculate the texture gradient
    - calculate color and texture histograms
    - Store all the necessary values in R.
    '''
    R = {}
    img_hsv=skimage.color.rgb2hsv(img[:,:,:-1])
    mask = img[:,:,-1]
    import matplotlib.pyplot as plt

    img_text = calc_texture_gradient(img[:,:,:-1])
    ### YOUR CODE HERE ###
    for i in range(np.max(mask)):
        y_all,x_all = np.where(mask==i)
        currentMask = mask==i
        R[i] = {
                    "labels": i,
                    "min_x": np.min(x_all),
                    "max_x": np.max(x_all),
                    "min_y": np.min(y_all),
                    "max_y": np.max(y_all),
                    "colorHist": calc_colour_hist(img_hsv[currentMask]),
                    "textureHist": calc_texture_hist(img_text[currentMask]),
                    "size": len(img_hsv[currentMask])
                }
    return R

def extract_neighbours(regions):

    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    # Hint 1: List of neighbouring regions
    # Hint 2: The function intersect has been written for you and is required to check neighbours
    neighbours = []
    ### YOUR CODE HERE ###
    for r_a in regions.items():
        for r_b in regions.items():
            if intersect(r_a[1],r_b[1])and r_a[0]<r_b[0]:
                neighbours.append((r_a,r_b))
    return neighbours


def merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {}
    ### YOUR CODE HERE
    rt["size"] = new_size
    rt["max_x"] = max(r1["max_x"], r2["max_x"])
    rt["min_x"] = min(r1["min_x"], r2["min_x"])
    rt["max_y"] = max(r1["max_y"], r2["max_y"])
    rt["min_y"] = min(r1["min_y"], r2["min_y"])
    rt["colorHist"] = (r1["colorHist"] * r1["size"] + r2["colorHist"] * r2["size"]) / new_size
    rt["textureHist"] = (r1["textureHist"] * r1["size"] + r2["textureHist"] * r2["size"]) / new_size
    rt["labels"] = r1["labels"] + r2["labels"]
    return rt


def selective_search(image_orig, scale=1.0, sigma=0.8, min_size=50):
    '''
    Selective Search for Object Recognition" by J.R.R. Uijlings et al.
    :arg:
        image_orig: np.ndarray, Input image
        scale: int, determines the cluster size in felzenszwalb segmentation
        sigma: float, width of Gaussian kernel for felzenszwalb segmentation
        min_size: int, minimum component size for felzenszwalb segmentation

    :return:
        image: np.ndarray,
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions: array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''

    # Checking the 3 channel of input image
    assert image_orig.shape[2] == 3, "Please use image with three channels."
    imsize = image_orig.shape[0] * image_orig.shape[1]

    # Task 1: Load image and get smallest regions. Refer to `generate_segments` function.
    image = generate_segments(image_orig, scale, sigma, min_size)

    if image is None:
        return None, {}

    # Task 2: Extracting regions from image
    # Task 2.1-2.4: Refer to functions "sim_colour", "sim_texture", "sim_size", "sim_fill"
    # Task 2.5: Refer to function "extract_regions". You would also need to fill "calc_colour_hist",
    # "calc_texture_hist" and "calc_texture_gradient" in order to finish task 2.5.
    R = extract_regions(image)

    # Task 3: Extracting neighbouring information
    # Refer to function "extract_neighbours"
    neighbours = extract_neighbours(R)

    # Calculating initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = calc_sim(ar, br, imsize)

    # Hierarchical search for merging similar regions
    while S != {}:

        # Get highest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # Task 4: Merge corresponding regions. Refer to function "merge_regions"
        t = max(R.keys()) + 1.0
        R[t] = merge_regions(R[i], R[j])

        # Task 5: Mark similarities for regions to be removed
        ### YOUR CODE HERE ###
        temp = []
        for k in S.keys():
            if i in k or j in k:
                temp.append(k)

        # Task 6: Remove old similarities of related regions
        ### YOUR CODE HERE ###
        for k in temp:
            del S[k]

        # Task 7: Calculate similarities with the new region
        ### YOUR CODE HERE ###

        for (a, b) in temp:
            if (a, b) != (i, j):
                if a != i:
                    current = a
                elif b != j:
                    current = b
                S[(t, current)] = calc_sim(R[t], R[current], imsize)

    # Task 8: Generating the final regions from R
    regions = []
    ### YOUR CODE HERE ###
    for k, r in list(R.items()):
        regions.append({
            'rect': (
                r['min_x'], r['min_y'],
                r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels']
        })

    return image, regions


