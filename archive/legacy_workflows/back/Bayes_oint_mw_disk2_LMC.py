#!/usr/bin/env python
# coding: utf-8
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import table

import agama as ag
from _04_Agama_ import int_ag

from astropy import table, units, coordinates
from galpy.util.coords import galcenrect_to_vxvyvz as VXYZgc_VXYZsun
from galpy.util.coords import rectgal_to_sphergal as Vxyz_Vsph
from matplotlib import rcParams
rcParams["figure.dpi"] = 150
from tqdm import tqdm
rng = np.random.default_rng()

import os, sys
from velocs import  vcyl_gc_vxy_gc, vxy_gc_vR_gc
from velocs_nerr import vxyz_gc_vrtp_gc

import argparse
from Calculate_obs import calculate_vhist_weight, calculate_RzSB, calculate_vhist
from Read_obs import Read_obsSB, Read_obsvhist



def loglike_cal(rs_ori, theta_ori, v_ori,verr_ori, vr_m, nr,nr_s, ntheta,ntheta_s, nv, rbd, tbd, vbd):
    x_hist=   vbd[:-1] + (vbd[1]-vbd[0]) / 2

    LL_vr = 0

    for it in range(ntheta_s,ntheta):
        for ir in range(nr_s,nr):
            ksel = np.ravel(np.where((rs_ori > rbd[ir]) & (rs_ori < rbd[ir+1]) & (theta_ori > tbd[it]*3.14/180)\
                            & (theta_ori < tbd[it+1]*3.14/180) ))

            v_ksel = v_ori[ksel]
            verr_ksel = verr_ori[ksel]

            w_ksel = 1
            model_v_hist = vr_m[ir, it, :]

            LL = 0
            for ipart in range(np.size(ksel)):
                kernal = np.exp(- (v_ksel[ipart] - x_hist)**2 / (2 * verr_ksel[ipart]**2) )
                Likelihood = 1/(np.sqrt(2*np.pi) * verr_ksel[ipart]) * np.sum(model_v_hist * kernal ) / np.sum(model_v_hist)
                LL += np.log10(Likelihood)

            LL_vr += LL * w_ksel
    return LL_vr


def int_one_model(base_path,model, rho0, rs, phalo, qhalo, alpha_halo, beta_halo, gamma, dtfile):
    
#    print(rho0, rs, phalo, qhalo, alpha_halo, beta_halo, gamma)
    s_file_affix = 'rho0%4.3f_rs%4.3f_p%5.3f_q%5.3f_a%5.3f_b%5.3f_gamma%2.3f'%(rho0, rs, phalo, qhalo, alpha_halo, beta_halo, gamma)

    data = table.Table.read(dtfile, format='ascii')

    ag.setUnits(mass = 1., length = 1., velocity = 1.)
    r = np.linspace(0.0, 150, 2500)
    xyz = np.c_[r, r*0, r*0]

    # bar angle
#    bar_ang = 25
#    x, omg, frame = 10, '-40', '*'

    x, omg, frame = 10, '0', '*'
    # time of orbital integration, particles per orbit
    #t_p, tra = 10, 500
    t_p, tra = 5,500  # 5, 500
    npart = tra 

    # define the Gravitational potential
    bar_mass = 10.2
    disk_mass = 10.5
    
    #pot_d = ag.Potential(type = 'Disk', mass = np.power(10,disk_mass), scaleRadius = 3,
    #                      scaleHeight = -0.4, innerCutoffRadius = 0, sersicIndex=1) # thin disk # disk and bulge in Vasiliev2021
    #pot_b = ag.Potential(type='Spheroid', mass = np.power(10, bar_mass) ,\
    #                        alpha = 1, gamma=0, beta=1.8, scaleRadius = 0.2, outerCutoffRadius=1.8, cutoffStrength=2)
                            
    pot_FB = ag.Potential(type = 'Ferrers', mass = np.power(10, bar_mass), scaleRadius = 3.5,
                          axisRatioY  = 0.44, axisRatioZ  = 0.31)

    pot_THD = ag.Potential(type = 'MiyamotoNagai', mass = 6e9, scaleRadius = 2.0,
                          scaleHeight = 0.9) # thick disk

    pot_MD = ag.Potential(type = 'Disk', mass = np.power(10,disk_mass), scaleRadius = 2.6,
                          scaleHeight = 0.3, innerCutoffRadius = 7, sersicIndex=1) # thin disk
                            
    # free parameters:  rho0, rs, gamma, p, q, beta_halo, alpha_halo
    
    alpha,beta,rcut,xi = 1,3,500,5
    gamma_halo = 0
    # alpha_halo + gamma_halo determines angle between x and X, we thus always set gamma_halo =0

    pot_halo = ag.Potential(type='Spheroid', rho0 = np.power(10, rho0) ,\
            alpha = alpha, gamma=gamma, beta=beta, scaleRadius = np.power(10, rs), p=phalo, q=qhalo, outerCutoffRadius=rcut, cutoffStrength=xi)

#    dens_my2  = make_halo_density_rot(np.power(10,rho0), rs, alpha, beta, gamma, rcut, xi, phalo, qhalo, alpha_halo*math.pi/180, beta_halo*math.pi/180, gamma_halo)
#    pot_halo = ag.Potential(type="Multipole", symmetry = 't', density=dens_my2, lmax=6, mmax=6, gridSizeR=36, rmin=1e-3, rmax=500 )

    pot = ag.Potential(pot_FB, pot_THD, pot_MD, pot_halo)


    ic_ag = np.array([data[ 'x_gc'], data[ 'y_gc'], data[ 'z_gc'],
                  data['vx_gc'], data['vy_gc'], data['vz_gc']]).T
    ag_out = int_ag(ic = ic_ag, pot = pot, t_p = t_p, tra = tra, omg = False)

    #print(np.size(ag_out['orb_tl']))
    #integ_out = np.array(list(set(np.arange(len(data))) - set(ag_out['orb_tl'])))
    #obsid_out = np.unique(np.array(data[integ_out]['obsid']))
    #s_good = list(set(np.arange(len(data))) - set(np.where(np.in1d(data['obsid'], obsid_out))[0]))
    #data = data[s_good]  # store the orginal data with suceess orbit integration
    
    
    data_ag = data['w', 'met'][ag_out['orb_tl']]
    #data_ag['obsid'] = np.array(data_ag['obsid']).astype(np.int64)
    data_ag['w'    ] = np.array(data_ag['w'    ]).astype(np.float16)
    data_ag['met'  ] = np.array(data_ag['met'  ]).astype(np.float16)
    data_ag['E'] = np.int32(ag_out['E'])
    #del ag_out['orb_tl'],
    del ag_out['E']


# In[36]:
    data_ag['t' ] = np.float16(ag_out['t' ])
    data_ag['x' ] = np.float32(ag_out['x' ])
    data_ag['y' ] = np.float32(ag_out['y' ])
    data_ag['z' ] = np.float32(ag_out['z' ])

    data_ag['vx'] = np.float32(ag_out['vx'])
    data_ag['vy'] = np.float32(ag_out['vy'])
    data_ag['vz'] = np.float32(ag_out['vz'])

    del ag_out['t'], ag_out['x' ], ag_out['y' ], ag_out['z' ], ag_out['vx'], ag_out['vy'], ag_out['vz']

########## v_spherical ##########
    data_ag['vr'], data_ag['v_phi'], data_ag['v_the'], data_ag['r3d'], data_ag['phi'], data_ag['theta'] = vxyz_gc_vrtp_gc(data_ag['x' ],   data_ag['y'],  data_ag['z'],
                    data_ag['vx'],  data_ag['vy'],  data_ag['vz'])


   # data_ag.write(base_path + model +r'/Orbits/orbits_'+s_file_affix + '.fits', overwrite = True)



#############################################
   # Grid parameter
    nRz = 25
    Rzmax = 50

    nv = 201
    vbd =  np.linspace(-800, 800, nv+1)

    rbd = np.array([4, 6, 8, 10, 12, 15, 20, 30, 50])
    nr = int(np.size(rbd)-1)

    tbd = np.array([0,15,30,45,60,90])
    ntheta = int(np.size(tbd)-1)


    # density distribution from observations.
    dtfile = base_path + '/data_for_model/lamost_dr8_SFlast_cut4_NS_LMCc/SB_Rz_25_cleanrG12-30.txt'
    den_Rz, den_err = Read_obsSB(dtfile, nRz)

    # calculate SB from the model
    z2d, R2d, den_m = calculate_RzSB(data_ag, nRz, Rzmax)
    den_m = den_m/(npart/2.)  # npart are in the full sky, npart/2 in the north hemisphere

    r3d_bin = np.sqrt(R2d**2 + z2d**2)
    uR_max = 40
    uR_min = 15
    

    den_err[den_err==0] = 1e9 # minimize the effect of this bin
    den_diff= (den_Rz - den_m)**2/ den_err**2
    den_diff = den_diff[(~np.isnan(den_diff)) & (z2d>2) & (den_Rz > 0) &(r3d_bin>uR_min) & (r3d_bin< uR_max)]
    chi2_SB = np.sum( den_diff )

    #t= table.Table()
    #t['density'] = den_m.flatten()
    #t['R2d'] = R2d.flatten()
    #t['z2d'] = z2d.flatten()

    #t.write(base_path + model + r'/SB_Rz/dRz_'+s_file_affix+'.fits', format = 'ascii',overwrite=True)



    # calculate the velocity histograms from model
    vr_mw, vp_mw, vt_mw = calculate_vhist_weight(data_ag, rbd, tbd, vbd)

    #t= table.Table()
    #t['vr'] = vr_mw.flatten()
    #t['vp'] = vp_mw.flatten()
    #t['vt'] = vt_mw.flatten()
    #t.write(base_path + model + r'/vvhist/vv_hist_'+s_file_affix+'.fits', format = 'ascii',overwrite=True)

    # calculate LL in 1-nr bins
    nr_end = nr
    nr_s = 2
    ntheta_s = 0

    ll_vr_w = loglike_cal(data['r_gc'], data['theta'], data['vr_gc'], data['vr_err'], vr_mw, nr_end, nr_s, ntheta,ntheta_s, nv, rbd, tbd, vbd)

    ll_vp_w = loglike_cal(data['r_gc'], data['theta'], data['v_phi'], data['vphi_err'], vp_mw, nr_end,nr_s, ntheta,ntheta_s, nv, rbd, tbd, vbd)

    ll_vt_w = loglike_cal(data['r_gc'], data['theta'], data['v_the'], data['vthe_err'],vt_mw, nr_end,nr_s, ntheta,ntheta_s, nv, rbd, tbd, vbd)


    # Calculate the likelihood of data from South
    dtfile_s = base_path + '/data_for_model/lamost_dr8_SFlast_cut4_NS_LMCc/halo_clean_S.txt'

    data_s = table.Table.read(dtfile_s, format='ascii')
    ll_vr = loglike_cal(data_s['r_gc'], data_s['theta'], data_s['vr_gc'], data_s['vr_err'], vr_mw, nr_end, nr_s, ntheta,ntheta_s, nv, rbd, tbd, vbd)
    
    ll_vp = loglike_cal(data_s['r_gc'], data_s['theta'], data_s['v_phi'], data_s['vphi_err'], vp_mw, nr_end,nr_s, ntheta,ntheta_s, nv, rbd, tbd, vbd)

    ll_vt = loglike_cal(data_s['r_gc'], data_s['theta'], data_s['v_the'], data_s['vthe_err'],vt_mw, nr_end,nr_s, ntheta,ntheta_s, nv, rbd, tbd, vbd)


    ppfile = base_path + model +'/llp/'+s_file_affix +'.txt'
    df = open(ppfile,'w')
    df.write(("%6.3f" % rho0) + '  '+("%4.3f" % rs) + '  '+ ("%5.3f" % phalo) + '  '+ ("%5.3f" % qhalo)+  '  '+ ("%5.1f" % alpha_halo) +  '  '+ ("%5.1f" % beta_halo)  +  '  '+ ("%5.3f" % gamma) + '  ' + ("%8.5e" % ll_vr) + '  ' + ("%8.5e" % ll_vp) + '  ' + ("%8.5e" % ll_vt) + '  ' + ("%8.5e" % ll_vr_w) + '  ' + ("%8.5e" % ll_vp_w) + '  ' + ("%8.5e" % ll_vt_w) + '  ' + ("%8.5e" % chi2_SB) )
    

    ll_tot = ( ll_vr +  ll_vp +  ll_vt + ll_vr_w + ll_vp_w +  ll_vt_w) - 0.5* chi2_SB
    
    del data_ag, ag_out, data, data_s
    return ll_tot
    #-----------------------------------------------------
    

