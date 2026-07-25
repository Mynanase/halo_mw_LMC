#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import table
from glob import glob

#######################################################
def calculate_RzSB_4phi(dt, nRz, Rzmax, nphi):
    # calculate SB from the model
    
    phi_bin = np.linspace(-np.pi, np.pi, nphi+1)
    Rbin = np.linspace(0, Rzmax, nRz+1)
    Zbin = np.linspace(0, Rzmax, nRz+1)

    z2d = np.zeros([nRz,nRz,nphi])
    R2d = np.zeros([nRz,nRz,nphi])
    for i in range(nRz):
        R2d[:,i,:] = (Rbin[i] + Rbin[i+1])/2
        z2d[i,:,:] = (Zbin[i] + Zbin[i+1])/2

    
    ww_model = dt['w'] # dt['w0']*dt['ws']

    R_gc = np.sqrt( dt['x']**2 + dt['y']**2 )
    pos3d = np.array([dt['z'], R_gc,  dt['phi'] ])
    
    nd = np.size(dt)
    pos3d = np.reshape(pos3d, [3, nd]).T

    hist_model, edges = np.histogramdd(pos3d, weights = ww_model, bins=(Zbin, Rbin,  phi_bin))

    den_m = hist_model / (2*np.pi/nphi * R2d)
    return z2d, R2d, den_m



def calculate_vhist_weight_4phi(dt, rbd, tbd, pbd, vbd):
        
    nr = int(np.size(rbd)-1)
    ntheta = int(np.size(tbd)-1)
    nv = int(np.size(vbd)-1)
    nphi = int(np.size(pbd)-1)
        
    # calculate the velocity histograms
    vr_m = np.zeros([nr, ntheta, nphi, nv])
    vp_m = np.zeros([nr, ntheta, nphi, nv])
    vt_m = np.zeros([nr, ntheta, nphi, nv])

    r3d_int1 = dt['r3d']
    z_int1 = dt['z']
    theta_int1 = dt['theta'] *180 / np.pi
    phi_int1 = dt['phi'] *180/ np.pi
    
    ww_model =dt['w'] # dt['w0']*dt['ws']
        
    bin3d = (rbd, tbd, pbd, vbd)
    
    dd3d = (dt['r3d'], theta_int1, phi_int1, dt['vr'])
    vr_m,xyz = np.histogramdd(dd3d, weights= ww_model, bins=bin3d)
    
    dd3d = (dt['r3d'], theta_int1, phi_int1, dt['v_phi'])
    vp_m,xyz = np.histogramdd(dd3d, weights= ww_model, bins=bin3d)

    dd3d = (dt['r3d'], theta_int1, phi_int1, dt['v_the'])
    vt_m,xyz = np.histogramdd(dd3d, weights= ww_model, bins=bin3d)


    vm_norm = np.sum(vr_m, axis = 3)
    for br in range(0, nr):
        for bt in range(0, ntheta):
            for bp in range(0, nphi):
                vr_m[br, bt, bp, :] = vr_m[br, bt, bp, :]/ vm_norm[br, bt, bp]
                vp_m[br, bt, bp, :] = vp_m[br, bt, bp, :]/ vm_norm[br, bt, bp]
                vt_m[br, bt, bp, :] = vt_m[br, bt, bp, :]/ vm_norm[br, bt, bp]
            
    return vr_m, vp_m, vt_m


def calculate_vhist_4phi(dt, rbd, tbd, pbd, vbd):
        
    nr = int(np.size(rbd)-1)
    ntheta = int(np.size(tbd)-1)
    nv = int(np.size(vbd)-1)
    nphi = int(np.size(pbd)-1)
        
    # calculate the velocity histograms
    vr_m = np.zeros([nr, ntheta, nphi, nv])
    vp_m = np.zeros([nr, ntheta, nphi, nv])
    vt_m = np.zeros([nr, ntheta, nphi, nv])

    r3d_int1 = dt['r3d']
    z_int1 = dt['z']
    theta_int1 = dt['theta'] *180 / np.pi
    phi_int1 = dt['phi'] *180/ np.pi
    
    bin3d = (rbd, tbd, pbd, vbd)
    
    dd3d = (dt['r3d'], theta_int1, phi_int1, dt['vr'])
    vr_m,xyz = np.histogramdd(dd3d, bins=bin3d)
    
    dd3d = (dt['r3d'], theta_int1, phi_int1, dt['v_phi'])
    vp_m,xyz = np.histogramdd(dd3d, bins=bin3d)

    dd3d = (dt['r3d'], theta_int1, phi_int1, dt['v_the'])
    vt_m,xyz = np.histogramdd(dd3d, bins=bin3d)


    vm_norm = np.sum(vr_m, axis = 3)
    for br in range(0, nr):
        for bt in range(0, ntheta):
            for bp in range(0, nphi):
                vr_m[br, bt, bp, :] = vr_m[br, bt, bp, :]/ vm_norm[br, bt, bp]
                vp_m[br, bt, bp, :] = vp_m[br, bt, bp, :]/ vm_norm[br, bt, bp]
                vt_m[br, bt, bp, :] = vt_m[br, bt, bp, :]/ vm_norm[br, bt, bp]
            
    return vr_m, vp_m, vt_m


