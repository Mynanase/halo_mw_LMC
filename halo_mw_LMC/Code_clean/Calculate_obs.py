#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import table
from glob import glob

#######################################################
def calculate_RzSB(dt, nRz, Rzmax):
    # calculate SB from the model
    rbd_m = np.linspace(0, Rzmax, nRz+1)
    z2d = np.zeros([nRz,nRz])
    R2d = np.zeros([nRz,nRz])
    for i in range(nRz):
        R2d[:,i] = (rbd_m[i] + rbd_m[i+1])/2
        z2d[i,:] = (rbd_m[i] + rbd_m[i+1])/2


    R_int1 = np.sqrt(dt['x']*dt['x'] +  dt['y']*dt['y'])
    z_int1 = dt['z']
    n_dRz_int, edge_r, edge_r = np.histogram2d(z_int1, R_int1, bins = [rbd_m, rbd_m], weights= dt['w'])
    den_m = n_dRz_int / (2*np.pi * R2d)
    return z2d, R2d, den_m


def calculate_vhist(dt, rbd, tbd, vbd):
        
    nr = int(np.size(rbd)-1)
    ntheta = int(np.size(tbd)-1)
    nv = int(np.size(vbd)-1)
    
    # calculate the velocity histograms
    vr_m = np.zeros([nr, ntheta, nv])
    vp_m = np.zeros([nr, ntheta, nv])
    vt_m = np.zeros([nr, ntheta, nv])

    r3d_int1 = dt['r3d']
    z_int1 = dt['z']
    theta_int1 = dt['theta'] *180 / np.pi
    
    bin3d = (rbd, tbd, vbd)
    
    dd3d = (dt['r3d'], theta_int1, dt['vr'])
    vr_m,xyz = np.histogramdd(dd3d, bins=bin3d)
    
    dd3d = (dt['r3d'], theta_int1, dt['v_phi'])
    vp_m,xyz = np.histogramdd(dd3d, bins=bin3d)

    dd3d = (dt['r3d'], theta_int1, dt['v_the'])
    vt_m,xyz = np.histogramdd(dd3d, bins=bin3d)


    vm_norm = np.sum(vr_m, axis = 2)
    for br in range(0, nr):
        for bt in range(0, ntheta):
            vr_m[br, bt, :] = vr_m[br, bt, :]/ vm_norm[br, bt]
            vp_m[br, bt, :] = vp_m[br, bt, :]/ vm_norm[br, bt]
            vt_m[br, bt, :] = vt_m[br, bt, :]/ vm_norm[br, bt]
            
    return vr_m, vp_m, vt_m


def calculate_vhist_weight(dt, rbd, tbd, vbd):

    nr = int(np.size(rbd)-1)
    ntheta = int(np.size(tbd)-1)
    nv = int(np.size(vbd)-1)

    # calculate the velocity histograms
    vr_m = np.zeros([nr, ntheta, nv])
    vp_m = np.zeros([nr, ntheta, nv])
    vt_m = np.zeros([nr, ntheta, nv])

    r3d_int1 = dt['r3d']
    z_int1 = dt['z']
    theta_int1 = dt['theta'] *180 / np.pi

    bin3d = (rbd, tbd, vbd)

    dd3d = (dt['r3d'], theta_int1, dt['vr'])
    vr_m,xyz = np.histogramdd(dd3d, bins=bin3d, weights = dt['w'])

    dd3d = (dt['r3d'], theta_int1, dt['v_phi'])
    vp_m,xyz = np.histogramdd(dd3d, bins=bin3d, weights = dt['w'])

    dd3d = (dt['r3d'], theta_int1, dt['v_the'])
    vt_m,xyz = np.histogramdd(dd3d, bins=bin3d, weights = dt['w'])


    vm_norm = np.sum(vr_m, axis = 2)
    for br in range(0, nr):
        for bt in range(0, ntheta):
            vr_m[br, bt, :] = vr_m[br, bt, :]/ vm_norm[br, bt]
            vp_m[br, bt, :] = vp_m[br, bt, :]/ vm_norm[br, bt]
            vt_m[br, bt, :] = vt_m[br, bt, :]/ vm_norm[br, bt]

    return vr_m, vp_m, vt_m

