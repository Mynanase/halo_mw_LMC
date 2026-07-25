#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import table
from glob import glob

#######################################################
def Read_obsSB_v2_4phi(dtfile, nRz, nphi):
    dd_data = table.Table.read(dtfile, format='ascii')
    den_Rz = dd_data['den']
    den_err = dd_data['den_srr']
    R2d =dd_data['R2d']
    z2d =dd_data['z2d']
    den_Rz = np.reshape(den_Rz,[nRz, nRz, nphi])
    den_err = np.reshape(den_err,[nRz, nRz, nphi])
    R2d = np.reshape(R2d,[nRz, nRz, nphi])
    z2d = np.reshape(z2d,[nRz, nRz, nphi])
    return den_Rz, den_err, R2d, z2d


def Read_obsSB_4phi(dtfile, nRz, nphi):
    dd_data = table.Table.read(dtfile, format='ascii')
    den_Rz = dd_data['den']
    den_err = dd_data['den_srr']
    den_Rz = np.reshape(den_Rz,[nRz, nRz, nphi])
    den_err = np.reshape(den_err,[nRz, nRz, nphi])
    return den_Rz, den_err



def Read_obsvhist_4phi(dtfile, nr, ntheta, nphi, nv):

    vbd =  np.linspace(-500, 500, nv +1)
    rbd = np.linspace(5,50,nr+1)
    tbd = np.linspace(0,90,ntheta+1)
    pbd = np.linspace(-180, 180, nphi+1)


    t1= table.Table.read(dtfile, format ='ascii')
    vr_obs = t1['vr_mean']
    vr_obs = np.reshape(vr_obs,[nr, ntheta, nphi, nv])
    vr_err = t1['vr_err']
    vr_err = np.reshape(vr_err,[nr, ntheta, nphi, nv])

    vp_obs = t1['vphi_mean']
    vp_obs = np.reshape(vp_obs,[nr, ntheta, nphi, nv])
    vp_err = t1['vphi_err']
    vp_err = np.reshape(vp_err,[nr, ntheta, nphi, nv])

    vt_obs = t1['vtheta_mean']
    vt_obs = np.reshape(vt_obs,[nr, ntheta, nphi, nv])
    vt_err = t1['vtheta_err']
    vt_err = np.reshape(vt_err,[nr, ntheta, nphi, nv])
    

    v_norm = np.sum(vr_obs, axis = 3)
    for br in range(0, nr):
        for bt in range(0, ntheta):
            for bp in range(0, nphi):
                vr_obs[br, bt, bp, :] = vr_obs[br, bt, bp, :]/ v_norm[br, bt, bp]
                vp_obs[br, bt, bp, :] = vp_obs[br, bt, bp, :]/ v_norm[br, bt, bp]
                vt_obs[br, bt, bp, :] = vt_obs[br, bt, bp, :]/ v_norm[br, bt, bp]
                
    return vr_obs, vr_err, vp_obs,vp_err, vt_obs, vt_err

