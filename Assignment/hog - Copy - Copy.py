#!/usr/bin/env python3
# ECE 471/536: Assignment 3 submission template
# Ammiel V00896915


# Using "as" nicknames a library so you don't have to use the full name
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import argparse as ap
import pprint

pp = pprint.PrettyPrinter(indent=4)

# Prevents python3 from failing
TODO = None

# Epsilon
EPS = 1e-6

"""Extract Histogram of Gradient features
Parameters
----------
X : ndarray NxHxW array where N is the number of instances/images 
                              HxW is the image dimensions
Returns
    features : NxD narray contraining the histogram features (D = 2304) 
-------
"""
# 20 marks: Histogram of Gradients
# noinspection PyPep8Naming
def hog(x):
    # DEFINITION
    ImNum, Height, Width = x.shape
    I_dx = np.zeros(shape=x.shape)  # temporary variable
    I_dy = np.zeros(shape=x.shape)  # temporary variable
    I_Mag = np.zeros(shape=x.shape)
    I_Grad = np.zeros(shape=x.shape)

    # 2 marks: apply a Sobel filter for each gradient (x,y) of kernel size 1
    #         HINT: your gx/gy size should be NxHxW
    sobel_x = [[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]
    sobel_x = np.array(sobel_x)
    sobel_x = sobel_x / 8

    sobel_y = [[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]]
    sobel_y = np.array(sobel_y)
    sobel_y = sobel_y / 8

    image_itr = 0
    for images in x:
        dx = cv2.filter2D(images, -1, sobel_x)
        dy = cv2.filter2D(images, -1, sobel_y)

        I_dx[image_itr] = dx
        I_dy[image_itr] = dy

    #2 marks: Calculate the mag and unsigned angle in range ( [0-180) degrees) for each pixel
    #         mag, ang should have size NxHxW
        power_sum = pow(dx, 2) + pow(dy, 2)
        I_Mag[image_itr] = np.sqrt(power_sum) # square root of summed squares

        grad_I = np.zeros(shape=[Height, Width]) # hold current gradient information
        for x in range(0, Height):
            for y in range(0, Width):
                if dx[x, y] == 0:  # if division by zero
                    grad_I[x, y] = 90
                else:
                    div_arc = dy(x, y) / dx(x, y)
                    grad = abs(360 * np.arctan(div_arc) / (2 * np.pi))
                    # since angles sinusoidally repeat negatively after 180
                    # check to make sure the angle is between 0 and 180
                    if grad > 180:
                        grad = grad - 180
                    grad_I[x, y] = grad

        I_Grad[image_itr] = grad_I
        image_itr = image_itr + 1

    # 1 mark: Split orientation matrix/tensor into 8x8 cells
    #        HINT: matrix size should be Nx(number of cells)x8x8
    number_of_cells = round(Height / 8) * round(Width / 8)

    Split_Orientation = np.zeros(shape=[ImNum, number_of_cells, 8, 8])
    Split_Magnitude = np.zeros(shape=[ImNum, number_of_cells, 8*8])

    height_incremental_factor = round(Height / 8)
    width_incremental_factor = round(Width / 8)

    Orientation_8x8 = np.zeros(shape=[8, 8])  # temporary variable
    Magnitude_64 = np.zeros(shape=8*8)  # temporary variable

    for image_itr in range(0, ImNum):
        cell_num = 0
        for H_ITR in range(0, height_incremental_factor):
            height_lower_limit = 0 + (8 * H_ITR)
            height_upper_limit = 7 + (8 * H_ITR)
            if height_upper_limit > Height:
                height_upper_limit = Height

            for W_ITR in range(0, width_incremental_factor):
                width_lower_limit = 1 + (8 * W_ITR)
                width_upper_limit = 8 + (8 * W_ITR)
                if width_upper_limit > Width:
                    width_upper_limit = Width

                x_Orientation = I_Grad[image_itr]
                x_Magnitude = I_Mag[image_itr]
                itr_split_mag = 0
                for X in range(height_lower_limit, height_upper_limit):
                    for Y in range(width_lower_limit, width_upper_limit):
                        Orientation_8x8[X, Y] = x_Orientation[X, Y]

    #1 mark: Split magnitude matrix/tensor into 8x8 cells, flattened to 64
    #        HINT: matrix size should be Nx(number of cells)x(8*8)
                        Magnitude_64[itr_split_mag] = x_Magnitude[X, Y]
                        itr_split_mag = itr_split_mag + 1

                Split_Orientation[image_itr, cell_num] = Orientation_8x8
                Split_Magnitude[image_itr, cell_num] = Magnitude_64

                cell_num = cell_num + 1

    # 1 mark: create an array to hold the feature histogram for each 8x8 cell in a image
    #        HINT: the array should have 3 dimensions
    feature_histogram = np.zeros(shape=[ImNum, number_of_cells, 9])  # from 0 to 160: 9 bins

    # Loop through and for each cell calculate the histogram of gradients
    # Don't worry if this is very slow
    # 1 mark: Find the two closest bins based on orientation of the gradient for pixel j in cell i
    # 1 mark: Calculate the bin ratio (how much of the magnitude is added to each bin)
    # 5 marks: add the magnitude contribution to each bin, based on orientation overlap with the bin (bin ratio)
    #         HINT: consider the edge cases for the bins
    for image_itr in range(0, ImNum):
        for cell_num in range(0, number_of_cells):
            Mag_W = 0
            for cell_H in range(0, 8):
                for cell_W in range(0, 8):
                    # Get the orientation...
                    Bin = Split_Orientation[image_itr, cell_num, cell_H, cell_W]
                    # and its relative magnitude
                    Bin_Mag = Split_Magnitude[image_itr, cell_num, Mag_W]

                    # Floor and Ceil the orientation
                    # if it is within two bin, it is mirrored
                    # and the magnitude is distributed
                    Bin_Floor = math.floor(Bin / 20)
                    Floor_Mag = 1 - (abs(Bin - (20 * Bin_Floor)) / 20)
                    Floor_Mag = Bin_Mag * Floor_Mag

                    Bin_Ceil = math.ceil(Bin / 20)
                    Ceil_Mag = 1 - (abs(Bin - (20 * Bin_Ceil)) / 20)
                    Ceil_Mag = Bin_Mag * Ceil_Mag

                    # Check to make sure it stays within the 9 bins: 0 to 8
                    # i.e floor or ceil 9 will correspond to bin 0 and so on
                    if Bin_Floor > 8:
                        Bin_Floor = Bin_Floor - 9
                    if Bin_Ceil > 8:
                        Bin_Ceil = Bin_Ceil - 9

                    # Adds distributed magnitude to the corresponding feature histogram
                    histogram_floor = feature_histogram[image_itr, cell_num, Bin_Floor]
                    histogram_ceil = feature_histogram[image_itr, cell_num, Bin_Ceil]
                    feature_histogram[image_itr, cell_num, Bin_Floor] = histogram_floor +  Floor_Mag
                    feature_histogram[image_itr, cell_num, Bin_Ceil] = histogram_ceil + Ceil_Mag

                    Mag_W = Mag_W + 1

    # Normally, there is a window normalization step here, but we're going to ignore that.
    # yay! :D

    # 1 mark: Reshape the histogram so that its NxD where N is the number of instances/images i
    #        and D is all the histograms per image concatenated into 1D vector



    # return the histogram as your feature vector
    pass

# 5 marks: Split the input matrix A into cells
def split_into_cells(A,cell_size=8):
    """Split ndarray into smaller array
    Parameters
    ----------
    A : ndarray of size NxHxW
    cell : tuple with (h,w) for cell size 
    Returns
    -------
    ndarray of size Nx(cell_h*cell_w)x(cell_h*cell_w)
    """
    pass