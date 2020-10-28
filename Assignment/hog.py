#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
    N,H,W = X.shape
    gx = np.zeros(X.shape)
    gy = np.zeros(X.shape)
    mag = np.zeros(X.shape)
    angle = np.zeros(X.shape)
    #2 marks: apply a Sobel filter for each gradient (x,y) of kernel size 1
    #         HINT: your gx/gy size should be NxHxW
    for i in range(N):
        gx[i] = cv2.Sobel(X[i], cv2.CV_32F, 1, 0, ksize=1)
        gy[i] = cv2.Sobel(X[i], cv2.CV_32F, 0, 1, ksize=1)

    #2 marks: Calculate the mag and unsigned angle in range ( [0-180) degrees) for each pixel
    #         mag, ang should have size NxHxW
    for i in range(N):
        mag[i], angle[i] = cv2.cartToPolar(gx[i], gy[i], angleInDegrees=True)
        angle[i] = angle[i] % 180
    
    
    #1 mark: Split orientation matrix/tensor into 8x8 cells, flattened to 64
    #        HINT: matrix size should be Nx(number of cells)x(8x8)
        #1 mark: Split magnitude matrix/tensor into 8x8 cells, flattened to 64
    #        HINT: matrix size should be Nx(number of cells)x(8*8)
    s = (N,16,8,8)
    s2 = (N,16,64)
    k= 0
    cell_direction = np.zeros(s)
    cell_magnitude = np.zeros(s)
    flat_cell_direction = np.zeros(s2)
    flat_cell_magnitude = np.zeros(s2)
    hist_bins = np.array([20,40,60,80,100,120,140,160,180])
    cell_number = 16
    cell_size = 64
    for images in range(N): 
        for j in range(4):
            for i in range(4):
                k=i+j
                if(j==1):
                    k=4+i
                elif(j==2):
                    k=8+i
                elif(j==3):
                    k=12+i
        #print(k)
                cell_direction[images][k] = angle[images][i*8:i*8+8,j*8:j*8+8]
                cell_magnitude[images][k] = mag[images][i*8:i*8+8,j*8:j*8+8]    
                flat_cell_direction[images][k] = cell_direction[images][k].flatten()
                flat_cell_magnitude[images][k] = cell_magnitude[images][k].flatten()

    
    #1 mark: create an array to hold the feature histogram for each 8x8 cell in a image
    #        HINT: the array should have 3 dimensions 
    s3 = (N,flat_cell_direction.shape[1],hist_bins.size)
    HOG_cell_hist = np.zeros(s3)

    #Loop through and for each cell calculate the histogram of gradients
    #Don't worry if this is very slow 
   
    #1 mark: Find the two closest bins based on orientation of the gradient for pixel j in cell i

    #1 mark: Calculate the bin ratio (how much of the magnitude is added to each bin)

       
    #5 marks: add the magnitude contribution to each bin, based on orientation overlap with the bin (bin ratio)
    #         HINT: consider the edge cases for the bins
    cell_number = flat_cell_direction.shape[1]
    cell_size  = flat_cell_direction.shape[2]

    for images in range(N):
        for cell_n in range(cell_number):
            for current in range(cell_size):
                currentd = flat_cell_direction[images][cell_n][current]
                currentm = flat_cell_magnitude[images][cell_n][current]
                diff = np.abs(currentd - hist_bins)
                first_bin_idx = np.where(diff == np.min(diff))[0][0]
                if np.max(diff) > 160:
                    diff[0] = 20-diff[-1]
                temp = hist_bins[[(first_bin_idx-1)%hist_bins.size, (first_bin_idx+1)%hist_bins.size]]
                temp2 = np.abs(currentd - temp)
                if temp2[1] > 160:
                    temp2[1] = 20-diff[-1]
                res = np.where(temp2 == np.min(temp2))[0][0]
                if res == 0 and first_bin_idx != 0:
                    second_bin_idx = first_bin_idx-1
                else:
                    second_bin_idx = first_bin_idx+1
                if second_bin_idx == 9:
                    second_bin_idx = 0
                if np.min(diff) == 0:
                    HOG_cell_hist[images][cell_n,first_bin_idx]+= currentm
                else:
                    HOG_cell_hist[images][cell_n,first_bin_idx] = HOG_cell_hist[images][cell_n,first_bin_idx] + (diff[second_bin_idx]/20.00 * currentm)
                    HOG_cell_hist[images][cell_n,second_bin_idx] = HOG_cell_hist[images][cell_n,second_bin_idx] + (diff[first_bin_idx]/20.00 * currentm)
    #Normally, there is a window normalization step here, but we're going to ignore that.

    #1 mark: Reshape the histogram so that its NxD where N is the number of instances/images i
    #        and D is all the histograms per image concatenated into 1D vector
    s4 = (N,144)

    Flatten_HOG_cell_hist = np.zeros(s4)
    for i in range(N):
        Flatten_HOG_cell_hist[i] = HOG_cell_hist[i].flatten()
    #return the histogram as your feature vector
    return Flatten_HOG_cell_hist
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
    N,H,W = A.shape
    cell_H = H/cell_size
    cell_W = W/cell_size
    number_cells = cell_H*cell_W
    s = (N,number_cells,cell_size,cell_size)
    k= 0
    split_cells = np.zeros(s)
    for images in range(N): 
        for j in range(cell_H):
            for i in range(cell_W):
                k=i+(j*cell_W)
        #print(k)
                split_cells[images][k] = A[images][i*cell_size:i*cell_size+cell_size,j*cell_size:j*cell_size+cell_size] 
    return split_cells
    pass


# In[ ]:





# In[ ]:




