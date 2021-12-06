#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:36:10 2021

@author: nikolaj
"""

import matplotlib.pyplot as plt
import scipy
import numpy as np
from sys import stdout
from scipy.special import comb


def progress(string):
    stdout.write("\r"+string)
    stdout.flush()

def imshow(im, cbar = False, vmin = None, vmax = None, cmap = 'viridis', cbar_name = None, hide_ax = True):
    plt.imshow(im, cmap = cmap, vmin = vmin, vmax = vmax, interpolation = 'none')
    if hide_ax: plt.axis('off')
    if cbar: plt.colorbar(label = cbar_name)
    plt.show()

def sft(im): return np.fft.fftshift(im, axes = (-1, -2))
def fft(im): return scipy.fft.fft2(im)
def ift(im): return scipy.fft.ifft2(im)
def fft2(im): return scipy.fft.rfft2(im)
def ift2(im): return scipy.fft.irfft2(im)

def eps(m): return (m == 0) + 1

def zernnoll2nm(j0, numskip):  #technically noll -1, so starting at j = 0

    j = j0 + numskip + 1

    indices = np.array(np.ceil((1+np.sqrt(1+8*j))/2),dtype=int)-1
    triangular_numbers = np.array(indices*(indices+1)/2).astype(int)
    n = indices -1

    r = j - triangular_numbers
    r +=n
    m = (-1)**j * ((n % 2) + 2 * np.array((r + ((n+1)%2))/2).astype(int))

    return n, m

def normalize(j, numskip = 3):
    n, m = zernnoll2nm(j, numskip)
    A = (2/eps(m))*(n+1)
    return np.float64(A)

def get_zern(dsize, pupilSize, pixelSize, num_c, rotang = 0, numskip = 3, p2 = False):
    
    zern = np.array([zernfun(i, dsize, pupilSize, pixelSize, rotang, numskip) 
                     for i in range(num_c)])
    
    return (zern,) + pupil(dsize, pupilSize, pixelSize, rotang, p2)

def zernfun(j, dim, pupilSize, pixelSize, rotang = 0, numskip = 3):
    
    r, theta, inds = pupil(dim, pupilSize, pixelSize, rotang)
    n, m = zernnoll2nm(j, numskip)

    Rmn = zern_r(n, m, r)
    ang = zern_theta(m, theta)
    
    zern = Rmn*ang
        
    return zern

def zern_theta(m, theta):
   
    if m<0: return np.sin(-m*theta)
    return np.cos(m*theta)

def zern_r(n, m, r):
    
    zum = np.zeros(r.shape)
    mn = int((n-np.abs(m))/2)    
    for k in range(mn+1):
        Rmn = (-1)**k * comb(n-k, k) * comb(n-2*k, mn-k) * r**(n-2*k)
        zum += Rmn
        
    return zum

def grid(dsize, pupilSize, pixelSize, rotang):
    
    d = dsize
    
    pixelSizePhaseY = 1/(d*pixelSize)
    yScale = pixelSizePhaseY/pupilSize
    y0 = d/2-0.5
    yi = np.linspace(-y0, y0, d)*yScale
    
    pixelSizePhaseX = 1/(d*pixelSize)
    xScale = pixelSizePhaseX/pupilSize
    x0 = d/2-0.5
    xi = np.linspace(-x0, x0, d)*xScale
    
    X,Y = np.meshgrid(xi, yi)
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y,X)+np.deg2rad(rotang)
    
    return r, theta

def pupil(dim, pupilSize, pixelSize, rotang, p2 = False):
    
    r, theta = grid(dim, pupilSize, pixelSize, rotang)
    P = r<1
    P2 = r>=1.9
    inds = np.nonzero(P)
    inds2 = np.nonzero(P2)
    if p2 == False: return r[inds], theta[inds], inds
    else: return r[inds], theta[inds], inds, inds2
