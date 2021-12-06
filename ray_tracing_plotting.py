#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:59:13 2021

@author: nikolaj
"""

import numpy as np
import matplotlib.pyplot as plt
from ray_tracing_functions import get_angles
from numba import njit
from tqdm import tqdm

def plot_single_ray(rs, N):
    fig, ax = plt.subplots()
    ax.plot(rs[:,1], rs[:,2], linewidth = .3, color = 'r')
    N_im = ax.imshow(N[0].T, origin = 'lower')
    fig.colorbar(N_im, label = "RI")
    ax.set_xlim([0,128])
    ax.set_ylim([0,128])
    ax.set_xlabel("Y")
    ax.set_ylabel("Z")
    ax.set_aspect(1)
    
    
            
def get_wavefront(pixelSize, l2p, Sk0, r0, base_N, max_angle, Ss, ps, rs, h, res = 128):
    
    wavefront = np.zeros((res, res), dtype = np.complex128)

    angles = get_angles(max_angle, res = res, extra = 0, return_grid = True)
    
    kx = angles[:,:,0]
    ky = angles[:,:,1]
    kz = angles[:,:,2]

    get_wavefront_loop(wavefront, r0, base_N, l2p, pixelSize, 
                       kx, ky, h, Ss, ps, rs)        
    
    wavefront *= 1/np.sqrt(kz)
    
    theta = np.arccos(kz)
    
    piston = np.angle(wavefront).mean()
    wavefront *= np.exp(-1j*piston)
    wavefront[theta>np.deg2rad(max_angle/2)] = np.nan
    
    normalize_factor = np.sqrt(np.nanmean(np.abs(wavefront)**2)/Sk0)
    wavefront /= normalize_factor
    
    return wavefront

def show_distortion(rs, pupil_loc, pupil_rad):
    

    x = rs[:,:,0].flatten()
    y = rs[:,:,1].flatten()
    
    fig, ax = plt.subplots()
    
    ax.scatter(x, y, edgecolor = 'none', facecolor = 'red', s = 1)


    px = pupil_loc[0]
    py = pupil_loc[1]
    pr = pupil_rad

    #plot pupil boundaries
    ax.plot([px-pr, px+pr], [py-pr, py-pr], 'k--')
    ax.plot([px-pr, px+pr], [py+pr, py+pr], 'k--')
    ax.plot([px-pr, px-pr], [py-pr, py+pr], 'k--')
    ax.plot([px+pr, px+pr], [py-pr, py+pr], 'k--')

    ax.set_aspect(1)    
    fig.show()

@njit(cache = True)
def ep(h, x, y):
    return np.maximum(1-(x/h)**2-(y/h)**2, 0)

def get_wavefront_loop(wavefront, r0, base_N, l2p, pixelSize, kx, ky, h, Ss, ps, rs):
    num_rays = len(Ss)
    for i in tqdm(range(num_rays), desc = "Calculating Wavefront"):
        update_wavefront(i, wavefront, r0, base_N, l2p, pixelSize, kx, ky, h, Ss, ps, rs)
        
@njit(cache = True)
def update_wavefront(i, wavefront, r0, base_N, l2p, pixelSize, kx, ky, h, Ss, ps, rs):
        path_length_offset = np.sqrt(np.sum((r0-rs[i])**2))*base_N
        path_length = max(ps[i]-path_length_offset, 0) #sometimes instabilities cause this value to be slightly less than 0, which is not physical
        path_length *= pixelSize*l2p

        wavefront += ep(h, kx-Ss[i][0], ky-Ss[i][1])*np.exp(1j*path_length)
