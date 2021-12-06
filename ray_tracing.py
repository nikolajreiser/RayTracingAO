#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:43:07 2021

@author: nikolaj
"""

import numpy as np
from functions import imshow, progress, ift, sft, fft2, ift2, get_zern, normalize
from ray_tracing_functions import trace_rays
from ray_tracing_plotting import get_wavefront, show_distortion
from ray_tracing_objects import  z_grad, worm1, worm_from_dat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import data


# N, base_N = worm_from_dat(t = '025')
N, base_N = worm1()
# N, base_N = z_grad()
x_size, y_size, z_size = N.shape
max_angle = 64.5

NA = 1.2
l = .532
l2p = 2*np.pi/l #length to phase
pixelSize = .096
pupilSize = NA/l
Sk0 = np.pi*(pixelSize*pupilSize)**2

num_c = 20
h_ep = 1.5e-2

#initial values
x0 = 70
y0 = 70
# z0 = z_size - 10
z0 = 5
r0 = np.array([x0, y0, z0], dtype = np.float64)


#trace rays 
Ss, ps, rs = trace_rays(r0, max_angle, N, res = 300, show_path = False)
wf = get_wavefront(pixelSize, l2p, Sk0, r0, base_N, max_angle, Ss, ps, rs, h_ep)

imshow(np.angle(wf), True, cbar_name = "Wavefront Error (rad)", cmap = 'twilight_shifted', vmin = -np.pi, vmax = np.pi)
# imshow(np.angle(wf), cmap = 'twilight_shifted', vmin = -np.pi, vmax = np.pi)

# wf /= np.abs(wf)
# normalize_factor = np.sqrt(np.nanmean(np.abs(wf)**2)/Sk0)
# wf /= normalize_factor

# wf *= np.exp(-1j*np.angle(wf))


# h = ift(np.nan_to_num(wf))
# s = sft(np.abs(h)**2)/Sk0
