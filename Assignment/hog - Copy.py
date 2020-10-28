#!/usr/bin/env python3
# ECE 471/536: Assignment 3 submission template


#Using "as" nicknames a library so you don't have to use the full name
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse as ap
import pprint

pp = pprint.PrettyPrinter(indent=4)

#Prevents python3 from failing
TODO = None

#Epsilon 
EPS=1e-6

"""Extract Histogram of Gradient features

Parameters
----------
X : ndarray NxHxW array where N is the number of instances/images 
                              HxW is the image dimensions

Returns
    features : NxD narray contraining the histogram features (D = 144) 
-------

"""
#20 marks: Histogram of Gradients
def hog(X):

    #2 marks: apply a Sobel filter for each gradient (x,y) of kernel size 1
    #         HINT: your gx/gy size should be NxHxW
    gx = cv2.Sobel(X, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(X, cv2.CV_32F, 0, 1, ksize=1)

    #2 marks: Calculate the mag and unsigned angle in range ( [0-180) degrees) for each pixel
    #         mag, ang should have size NxHxW
    
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    #1 mark: Split orientation matrix/tensor into 8x8 cells, flattened to 64
    #        HINT: matrix size should be Nx(number of cells)x(8x8)
    ts = template.size #Number of elements in the template (3).
    #New padded array to hold the resultant gradient image.
    new_img = numpy.zeros((img.shape[0]+ts-1, 
                           img.shape[1]+ts-1))
    new_img[numpy.uint16((ts-1)/2.0):img.shape[0]+numpy.uint16((ts-1)/2.0), 
    numpy.uint16((ts-1)/2.0):img.shape[1]+numpy.uint16((ts-1)/2.0)] = img
    result = numpy.zeros((new_img.shape))
    
    for r in numpy.uint16(numpy.arange((ts-1)/2.0, img.shape[0]+(ts-1)/2.0)):
        for c in numpy.uint16(numpy.arange((ts-1)/2.0, 
                              img.shape[1]+(ts-1)/2.0)):
            curr_region = new_img[r-numpy.uint16((ts-1)/2.0):r+numpy.uint16((ts-1)/2.0)+1, 
                                  c-numpy.uint16((ts-1)/2.0):c+numpy.uint16((ts-1)/2.0)+1]
            curr_result = curr_region * template
            score = numpy.sum(curr_result)
            result[r, c] = score
    #Result of the same size as the original image after removing the padding.
    result_img = result[numpy.uint16((ts-1)/2.0):result.shape[0]-numpy.uint16((ts-1)/2.0), 
                        numpy.uint16((ts-1)/2.0):result.shape[1]-numpy.uint16((ts-1)/2.0)]
    #1 mark: Split magnitude matrix/tensor into 8x8 cells, flattened to 64
    #        HINT: matrix size should be Nx(number of cells)x(8*8)
    
    #1 mark: create an array to hold the feature histogram for each 8x8 cell in a image
    #        HINT: the array should have 3 dimensions 
    
    #Loop through and for each cell calculate the histogram of gradients
    #Don't worry if this is very slow 
   
    #1 mark: Find the two closest bins based on orientation of the gradient for pixel j in cell i

    #1 mark: Calculate the bin ratio (how much of the magnitude is added to each bin)

       
    #5 marks: add the magnitude contribution to each bin, based on orientation overlap with the bin (bin ratio)
    #         HINT: consider the edge cases for the bins
    horizontal_gradient_square = numpy.power(horizontal_gradient, 2)
    vertical_gradient_square = numpy.power(vertical_gradient, 2)
    sum_squares = horizontal_gradient_square + vertical_gradient_square
    grad_magnitude = numpy.sqrt(sum_squares)
    return grad_magnitude
    #Normally, there is a window normalization step here, but we're going to ignore that.

    #1 mark: Reshape the histogram so that its NxD where N is the number of instances/images i
    #        and D is all the histograms per image concatenated into 1D vector

    #return the histogram as your feature vector
    pass

#5 marks: Split the input matrix A into cells
def split_into_cells(A,cell_size=8):
    """Split ndarray into smaller array
	
    Parameters
    ----------
    A : ndarray of size NxHxW
    cell : tuple with (h,w) for cell size 

    Returns
    -------
    ndarray of size Nx(number of cells)x(cell_size*cell_size)
    """
    pass
