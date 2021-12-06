#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:06:40 2021

@author: nikolaj
"""
import numpy as np
from skimage.draw import ellipsoid
from numpy.random import seed, rand, randint
from skimage.io import imread
from skimage.filters import gaussian
import scipy
from skimage.morphology import binary_dilation

def z_grad():
    x_size = 512
    y_size = 512
    z_size = 128
    
    base_N = 1.33
    N = np.ones((x_size, y_size, z_size))*base_N
    grad = np.arange(z_size)/z_size
    N = N + 10*grad[None, None, :]
    
    return N, base_N

def worm1():
    dsize = 256
    x_size = dsize
    y_size = dsize
    z_size = dsize
    
    seed(0)
    
    #c elegans worm is about 40x40 um, so each voxel is roughly .1x.1x.1 um
    #immersion fluid has RI 1.33 (maybe)
    #cells have a RI of 1.36-1.39 and a radius of .1-2 um
    
    base_N = 1.33
    N = np.ones((x_size, y_size, z_size))*base_N

    spacing = (1, 1, 1)
    num_cells = 2000
    
    cell_min_size = 1
    cell_max_size = 10
    cell_range_size = cell_max_size-cell_min_size
    cell_min_RI = 1.34
    cell_max_RI = 1.39
    cell_range_RI = cell_max_RI-cell_min_RI

        
    for i in range(num_cells):
        
        cell_size = rand(1)*(cell_range_size)+cell_min_size
        cell_RI = rand(1)*(cell_range_RI)+cell_min_RI
        cl = randint(cell_max_size*2, dsize - cell_max_size*2, size = 3)
        
        cell_inds = ellipsoid(cell_size, cell_size, cell_size, spacing = spacing)
        cs = cell_inds.shape
        cell = np.zeros(cs)
        cell[cell_inds == True] = cell_RI
        cell[cell_inds == False] = base_N
        N[cl[0]:cl[0]+cs[0],cl[1]:cl[1]+cs[1],cl[2]:cl[2]+cs[2]] = np.maximum(N[cl[0]:cl[0]+cs[0],cl[1]:cl[1]+cs[1],cl[2]:cl[2]+cs[2]], cell)
                
        
    return N, base_N

def worm_from_dat(t = '025'):
    base_N = 1.33

    worm = imread(f'/home/nikolaj/Downloads/Annotations_share/t{t}_seg.tif')
    worm = np.float64(worm)
    worm[worm != 0] = 1
    
    
    #first create shell
    initial_shell = flood_fill_hull(worm)
    num_dilations = 2
    shell_thickness = 1
    shell_RI = 1.37
    
    for i in range(num_dilations):
        initial_shell = binary_dilation(initial_shell)
    
    outer_shell = initial_shell[:]
    for i in range(shell_thickness):
        outer_shell = binary_dilation(outer_shell)
        
    shell = np.float64(outer_shell)-np.float64(initial_shell)
    shell *= shell_RI-base_N
    
    
    
    
    #the create cells
    RI = np.zeros(worm.shape)
    inds_cells = np.nonzero(worm)
    
    seed(0)
    
    cell_min_RI = 1.34
    cell_max_RI = 1.39
    cell_range_RI = cell_max_RI-cell_min_RI
    

    cell_RI = rand(len(inds_cells[0]))*(cell_range_RI)+cell_min_RI-base_N
    RI[inds_cells] = cell_RI
    
    
    RI = gaussian(RI, sigma = 1)
    RI *= worm
    
    RI += shell
    RI = trim_zeros(RI)
    RI += base_N
    

    return RI, base_N
    
    
    
def trim_zeros(arr):
    """Returns a trimmed view of an n-D array excluding any outer
    regions which contain only zeros.
    """
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(arr))
    return arr[slices]
    
def flood_fill_hull(image):    
    points = np.transpose(np.where(image))
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices]) 
    idx = np.stack(np.indices(image.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img
