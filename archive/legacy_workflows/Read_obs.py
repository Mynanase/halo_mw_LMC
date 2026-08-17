#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import table
from glob import glob

#######################################################
def Read_obsSB_v2(dtfile, nRz):
    dd_data = table.Table.read(dtfile, format='ascii')
    den_Rz = dd_data['den']
    den_err = dd_data['den_srr']
    R2d =dd_data['R2d']
    z2d =dd_data['z2d']
    den_Rz = np.reshape(den_Rz,[nRz, nRz])
    den_err = np.reshape(den_err,[nRz, nRz])
    R2d = np.reshape(R2d,[nRz, nRz])
    z2d = np.reshape(z2d,[nRz, nRz])
    return den_Rz, den_err, R2d, z2d


def Read_obsSB(dtfile, nRz):
    dd_data = table.Table.read(dtfile, format='ascii')
    den_Rz = dd_data['den']
    den_err = dd_data['den_srr']
    den_Rz = np.reshape(den_Rz,[nRz, nRz])
    den_err = np.reshape(den_err,[nRz, nRz])
    return den_Rz, den_err



def Read_obsvhist(dtfile, nr, ntheta, nv):

    vbd =  np.linspace(-500, 500, nv +1)
    rbd = np.linspace(5,50,nr+1)
    tbd = np.linspace(0,90,ntheta+1)

    t1= table.Table.read(dtfile, format ='ascii')
    vr_obs = t1['vr_mean']
    vr_obs = np.reshape(vr_obs,[nr, ntheta, nv])
    vr_err = t1['vr_err']
    vr_err = np.reshape(vr_err,[nr, ntheta, nv])

    vp_obs = t1['vphi_mean']
    vp_obs = np.reshape(vp_obs,[nr, ntheta, nv])
    vp_err = t1['vphi_err']
    vp_err = np.reshape(vp_err,[nr, ntheta, nv])

    vt_obs = t1['vtheta_mean']
    vt_obs = np.reshape(vt_obs,[nr, ntheta, nv])
    vt_err = t1['vtheta_err']
    vt_err = np.reshape(vt_err,[nr, ntheta, nv])

#    v_R_obs = t1['v_R_mean']
#    v_R_obs = np.reshape(v_R_obs,[nr, ntheta, nv])
#    v_R_err = t1['v_R_err']
#    v_R_err = np.reshape(v_R_err,[nr, ntheta, nv])


#    vz_obs = t1['vz_mean']
#    vz_obs = np.reshape(vz_obs,[nr, ntheta, nv])
#    vz_err = t1['vz_err']
#    vz_err = np.reshape(vz_err,[nr, ntheta, nv])

    v_norm = np.sum(vr_obs, axis = 2)

    for br in range(0, nr):
        for bt in range(0, ntheta):
            vr_obs[br, bt, :] = vr_obs[br, bt, :]/ v_norm[br, bt]
            vr_err[br, bt, :] = vr_err[br, bt, :]/ v_norm[br, bt]

            vp_obs[br, bt, :] = vp_obs[br, bt, :]/ v_norm[br, bt]
            vp_err[br, bt, :] = vp_err[br, bt, :]/ v_norm[br, bt]

            vt_obs[br, bt, :] = vt_obs[br, bt, :]/ v_norm[br, bt]
            vt_err[br, bt, :] = vt_err[br, bt, :]/ v_norm[br, bt]

 #           v_R_obs[br, bt, :] = v_R_obs[br, bt, :]/ v_norm[br, bt]
 #           v_R_err[br, bt, :] = v_R_err[br, bt, :]/ v_norm[br, bt]

 #           vz_obs[br, bt, :] = vz_obs[br, bt, :]/ v_norm[br, bt]
 #           vz_err[br, bt, :] = vz_err[br, bt, :]/ v_norm[br, bt]


    return vr_obs, vr_err, vp_obs,vp_err, vt_obs, vt_err

