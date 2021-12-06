#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 11:13:08 2021

@author: nikolaj
"""
import numpy as np
from numpy.linalg import norm
from numba import njit, prange
from functions import progress, fft, ift
from tqdm import tqdm

@njit(cache = True)
def trace_ray(r, S, ss, N, V, dN_mag):
    tol = 1e-8
    min_ss = .05
    max_iter = 1e6
    i = 0
    p = 0
    dims = np.array(N.shape, dtype = np.float64)
    
    while i < max_iter:
            
        # make sure ray is still in box
        if (r<0.0).any(): break
        if (r>dims).any(): break
        
        #get pixel location of current ray
        l0 = int(r[0])
        l1 = int(r[1])
        l2 = int(r[2])

        #calculate quantities
        V0 = V[:,l0, l1, l2]
        dN_mag0 = dN_mag[l0, l1, l2]
        N0 = N[l0, l1, l2]
        
        
        #check if gradient is constant
        if dN_mag0 == 0:
            r = r + ss*S
            p = p + ss*N0
            c = 0
            
        else:
            
            c = np.sum(S*V0) #cosine of incident angle
            if np.abs(c) > 1: #sometimes c is slightly larger than 1
                c = np.sign(c)
            s = np.sqrt(1-c**2) #choose positive sine
            
            v = max(np.abs(c*ss), min_ss)
            if np.abs(c) <= tol: #if c is close to 0, ray is perpendicular to RI gradient
                v = min_ss
            else:
                v *= np.sign(c)
            
            
            #check if RI gradient is in opposite direction of ray
            if c < -tol:
                
                #in this case, forward direction of ray is in negative direction for v
                #first check maximum distance ray travels before being turned around
                
                vmax = N0*(s-1)/dN_mag0
                v = max(v, vmax) #chooses less negative (so smaller abs value) step size

            
            D0 = dN_mag0/N0
                
            Y = np.sqrt((1+D0*v)**2-s**2)

            if np.abs(c) > tol: #if c is close to 0, ray is perpendicular to RI gradient
                Y *= np.sign(c)

            # check if RI gradient is exactly parallel to ray
            if np.abs(s) > tol:
                
                W = np.cross(np.cross(V0, S), V0)/s #normalize W
                Z = np.log((1+D0*v)*(1+Y/(1+D0*v))/(1+c))

                # if np.abs(c) > tol: #if c is close to 0, ray is perpendicular to RI gradient
                #     W = np.sign(c)*W/norm(W)
                #update ray
                S = V0*Y/(1+D0*v) + W*s/(1+D0*v)
                r = r + V0*v + W*s/D0*Z
                p = p + N0/(2*D0) * (s**2*Z + Y*(1+D0*v) - c)
            
            else:
                
                #update ray- ray stays in same direction, but travels longer path
                S = V0*Y/(1+D0*v)
                r = r+V0*v
                p = p + N0/(2*D0) * (Y*(1+D0*v) - c)

            if np.any(r == np.nan): break
            if np.any(S == np.nan): break
            if p == np.nan: break
        
        i += 1
        
    return S, p, r

def get_N_info(N):
    V = np.array(np.gradient(N))
    dN_mag = norm(V, axis = 0)
    V = np.divide(V, dN_mag, where = (dN_mag != 0))    
    return dN_mag, V


def trace_rays(r0, max_angle, N, res = 100, show_path = False):
    
    ss = .1 #step size should always be less than 1, can probably get away with .5 if performance is needed
    
    #make sure that all N values are valid
    if np.any(N<1):
        print("RI values less than 1")
        return -1
    
    dN_mag, V = get_N_info(N)
    Ss = get_angles(max_angle, res)
    
    num_rays = len(Ss)
    S_out = np.zeros((num_rays, 3))
    p_out = np.zeros((num_rays))
    r_out = np.zeros((num_rays, 3))
    
    if show_path:
        
        r_path = []
        S_path = []
        p_path = []
        c_path = []
        N_path = []
        dN_mag_path = []
        V_path = []
        
        trace_rays_loop_with_path(r0, Ss, ss, res, N, V, dN_mag, S_out, p_out, r_out, r_path, S_path, p_path, c_path, N_path, dN_mag_path, V_path)
        return S_out, p_out, r_out, r_path, S_path, p_path, c_path, N_path, dN_mag_path, V_path
        
    trace_rays_loop(r0, Ss, ss, res, N, V, dN_mag, S_out, p_out, r_out)
            
    return S_out, p_out, r_out

# @njit(cache = True, parallel = True)
def trace_rays_loop(r0, Ss, ss, res, N, V, dN_mag, S_out, p_out, r_out):
    num_rays = len(Ss)
    for i in tqdm(range(num_rays), desc = "Tracing Rays"):
        # if i%1000 == 0: progress(str((num_rays-i)//1000))
        S_out[i], p_out[i], r_out[i] = trace_ray(r0[:], Ss[i], ss, N, V, dN_mag)
    
    
def trace_rays_loop_with_path(r0, Ss, ss, res, N, V, dN_mag, S_out, p_out, r_out, r_path, S_path, p_path, c_path, N_path, dN_mag_path, V_path):
    num_rays = len(Ss)
    for i in range(num_rays):
        progress(str(num_rays-i))
        S_out[i], p_out[i], r_out[i], r_temp, S_temp, p_temp, c_temp, N_temp, dN_mag_temp, V_temp = trace_ray_with_path(r0[:], Ss[i], ss, N, V, dN_mag)
        r_path.append(r_temp)
        S_path.append(S_temp)
        p_path.append(p_temp)
        c_path.append(c_temp)
        N_path.append(N_temp)
        dN_mag_path.append(dN_mag_temp)
        V_path.append(V_temp)
        
def get_angles(max_angle, res = 100, extra = 10, return_grid = False):
    #return square pupil instead of circle since it's easier
    
    angle = np.deg2rad(extra+max_angle/2)
    sin_theta_max = np.sin(angle) 
    sin_theta_vals =  np.linspace(sin_theta_max, -sin_theta_max, res, True)

    x, y = np.meshgrid(sin_theta_vals, sin_theta_vals)    

    angles = np.zeros((res, res, 3))
    angles[:,:,0] = x
    angles[:,:,1] = y
    angles[:,:,2] = np.sqrt(1-(x**2+y**2))
    
    if return_grid: return angles
    
    #reject angles greater than cos_theta_max
    angles = angles.reshape((res*res, 3))
    theta = np.arccos(angles[:,2])
    inds = np.nonzero(theta>angle)
    angles = np.delete(angles, inds, axis = 0) #delete keeps values at inds, and deletes the rest
    
    return angles
    

# jit doesn't help this one
# @njit(cache = True)
def trace_ray_with_path(r, S, ss, N, V, dN_mag):
    max_iter = np.int32(1e6)
    i = 0
    p = 0
    c = 0
    dims = np.array(N.shape, dtype = np.float64)
    tol = 1e-5
    min_ss = .05
    
    rs = np.zeros((max_iter, 3))
    Ss = np.zeros((max_iter, 3))
    ps = np.zeros((max_iter))
    cs = np.zeros((max_iter))
    N_path = np.zeros((max_iter))
    dN_mag_path = np.zeros((max_iter))
    V_path = np.zeros((max_iter, 3))

    while i < max_iter:
        
            
        # make sure ray is still in box
        if (r<0.0).any(): break
        if (r>dims).any(): break
        
        #get pixel location of current ray
        l0 = int(r[0])
        l1 = int(r[1])
        l2 = int(r[2])

        #calculate quantities
        V0 = V[:,l0, l1, l2]
        dN_mag0 = dN_mag[l0, l1, l2]
        N0 = N[l0, l1, l2]
        
        
        #check if gradient is constant
        if dN_mag0 == 0:
            r = r + ss*S
            p = p + ss*N0
            c = 0
            
        else:
            
            c = np.sum(S*V0) #cosine of incident angle
            s = np.sqrt(1-c**2) #choose positive sine
            
            v = max(np.abs(c*ss), min_ss)
            if np.abs(c) <= tol: #if c is close to 0, ray is perpendicular to RI gradient
                v = min_ss
            else:
                v *= np.sign(c)
            
            
            #check if RI gradient is in opposite direction of ray
            if c < -tol:
                
                #in this case, forward direction of ray is in negative direction for v
                #first check maximum distance ray travels before being turned around
                
                vmax = N0*(s-1)/dN_mag0
                v = max(v, vmax) #chooses less negative (so smaller abs value) step size

            
            D0 = dN_mag0/N0
                
            Y = np.sqrt((1+D0*v)**2-s**2)

            if np.abs(c) > tol: #if c is close to 0, ray is perpendicular to RI gradient
                Y *= np.sign(c)

            # check if RI gradient is exactly parallel to ray
            if np.abs(s) > tol:
                
                W = np.cross(np.cross(V0, S), V0)/s #normalize W
                Z = np.log((1+D0*v)*(1+Y/(1+D0*v))/(1+c))

                # if np.abs(c) > tol: #if c is close to 0, ray is perpendicular to RI gradient
                #     W = np.sign(c)*W/norm(W)
                #update ray
                S = V0*Y/(1+D0*v) + W*s/(1+D0*v)
                r = r + V0*v + W*s/D0*Z
                p = p + N0/(2*D0) * (s**2*Z + Y*(1+D0*v) - c)
            
            else:
                
                #update ray- ray stays in same direction, but travels longer path
                S = V0*Y/(1+D0*v)
                r = r+V0*v
                p = p + N0/(2*D0) * (Y*(1+D0*v) - c)


        # record path
        rs[i] = r
        Ss[i] = S
        ps[i] = p
        cs[i] = c
        N_path[i] = N0
        dN_mag_path[i] = dN_mag0
        V_path[i] = V0

        i += 1
        
    #trim excess
    rs = rs[:i]
    Ss = Ss[:i]
    ps = ps[:i]
    cs = cs[:i]
    N_path = N_path[:i]
    dN_mag_path = dN_mag_path[:i]
    V_path = V_path[:i]
    
    return S, p, r, rs, Ss, ps, cs, N_path, dN_mag_path, V_path

def propagate(R_in, Theta_in, inds, E, d):
    
    angSpec = fft(E)
    R = np.zeros_like(E)
    Theta = np.zeros_like(E)
    R[inds] = R_in
    Theta[inds] = Theta_in
    
    
    n = 1.33
    NA = 1.2
    kx = R*np.cos(Theta)*NA/n
    ky = R*np.sin(Theta)*NA/n
    kz = np.sqrt(np.complex128(1-(kx**2+ky**2)))
    
    propagator = np.exp(1j*d*kz)
    
    angSpec *= propagator
    
    return ift(angSpec)

