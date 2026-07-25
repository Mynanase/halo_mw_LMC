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
from Calculate_obs_4phi import calculate_vhist_weight_4phi, calculate_RzSB_4phi, calculate_vhist_4phi
from Read_obs_4phi import Read_obsSB_4phi, Read_obsvhist_4phi

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def loglike_cal(rs_ori, theta_ori, phi_ori, v_ori,verr_ori, vr_m, nr,nr_s, ntheta,ntheta_s, nv, rbd, tbd, pbd, vbd):
    x_hist=   vbd[:-1] + (vbd[1]-vbd[0]) / 2

    LL_vr = 0
    
    nphi = np.size(pbd)-1
    
    for it in range(ntheta_s,ntheta):
        for ir in range(nr_s,nr):
            for iphi in range(nphi):
                ksel = np.ravel(np.where((rs_ori > rbd[ir]) & (rs_ori < rbd[ir+1]) & (theta_ori > tbd[it]*3.14/180)\
                            & (theta_ori < tbd[it+1]*3.14/180) & (phi_ori > pbd[iphi]*3.14/180)\
                            & (phi_ori < pbd[iphi+1]*3.14/180) ))

                v_ksel = v_ori[ksel]
                verr_ksel = verr_ori[ksel]
                model_v_hist = vr_m[ir, it, iphi, :]

                LL = 0
                if np.size(ksel)>=1:
                    for ipart in range(np.size(ksel)):
                        kernal = np.exp(- (v_ksel[ipart] - x_hist)**2 / (2 * verr_ksel[ipart]**2) )
                        #print(np.size(kernal), np.size(verr_ksel[ipart]), np.size(model_v_hist), np.size(np.sum(model_v_hist)))
                        Likelihood = 1/(np.sqrt(2*np.pi) * verr_ksel[ipart]) * np.sum(model_v_hist * kernal ) / np.sum(model_v_hist)
                        LL += np.log10(Likelihood)

                LL_vr += LL 
    return LL_vr


def int_one_model(base_path,model, rho0, rs, phalo, qhalo, alpha_halo, beta_halo, gamma, dtfile):
    
#    print(rho0, rs, phalo, qhalo, alpha_halo, beta_halo, gamma)
    s_file_affix = 'rho0%4.3f_rs%4.3f_p%5.3f_q%5.3f_a%5.3f_b%5.3f_gamma%2.3f'%(rho0, rs, phalo, qhalo, alpha_halo, beta_halo, gamma)

    data = table.Table.read(dtfile, format='ascii')


    ag.setUnits(length=1, velocity=1, mass=1)  # work in units of 1 kpc, 1 km/s, 1 Msun)
    t_p, tra = 10, 1000
    npart = tra

    # define the Gravitational potential
    bar_mass = 10.2
    disk_mass = 10.5
    
    pot_d = ag.Potential(type = 'Disk', mass = np.power(10,disk_mass), scaleRadius = 3,
                          scaleHeight = -0.4, innerCutoffRadius = 0, sersicIndex=1) # thin disk # disk and bulge in Vasiliev2021
    pot_b = ag.Potential(type='Spheroid', mass = np.power(10, bar_mass) ,\
                            alpha = 1, gamma=0, beta=1.8, scaleRadius = 0.2, outerCutoffRadius=1.8, cutoffStrength=2)
                            
    # free parameters:  rho0, rs, gamma, p, q, beta_halo, alpha_halo
    
    alpha,beta,rcut,xi = 1,3,500,5

    pot_halo = ag.Potential(type='Spheroid', rho0 = np.power(10, rho0) ,\
            alpha = alpha, gamma=gamma, beta=beta, scaleRadius = np.power(10, rs), p=phalo, q=qhalo, outerCutoffRadius=rcut, cutoffStrength=xi)
    print('rho0, rs, phalo, qhalo, gamma:', rho0, rs, phalo, qhalo, gamma)

########################
#    gamma_halo = 0
#    alpha_halo + gamma_halo determines angle between x and X, we thus always set gamma_halo =0

#    dens_my2  = make_halo_density_rot(np.power(10,rho0), rs, alpha, beta, gamma, rcut, xi, phalo, qhalo, alpha_halo*math.pi/180, beta_halo*math.pi/180, gamma_halo)
#    pot_halo = ag.Potential(type="Multipole", symmetry = 't', density=dens_my2, lmax=6, mmax=6, gridSizeR=36, rmin=1e-3, rmax=500 )

    pot = ag.Potential(pot_d, pot_b, pot_halo)

    ic_ag = np.array([data[ 'x_gc'], data[ 'y_gc'], data[ 'z_gc'],
                  data['vx_gc'], data['vy_gc'], data['vz_gc']]).T
    ag_out = int_ag(ic = ic_ag, pot = pot, t_p = t_p, tra = tra, omg = False)

    #print(np.size(ag_out['orb_tl']))
    #integ_out = np.array(list(set(np.arange(len(data))) - set(ag_out['orb_tl'])))
    #obsid_out = np.unique(np.array(data[integ_out]['obsid']))
    #s_good = list(set(np.arange(len(data))) - set(np.where(np.in1d(data['obsid'], obsid_out))[0]))
    #data = data[s_good]  # store the orginal data with suceess orbit integration
    
    
    data_ag = data['w0', 'w', 'met'][ag_out['orb_tl']]
    #data_ag['obsid'] = np.array(data_ag['obsid']).astype(np.int64)
    data_ag['w0'    ] = np.array(data_ag['w0'    ]).astype(np.float16)
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



   # calculate the weights of stars according to the imcomplete sptial coverage
#    nd = np.size(data)
#    norbit = int(np.size(data_ag)/tra)  # the number of orbit successfully integrated
#    if norbit != nd: print('The orbit number does not equal the number of stars')


#    rbin = np.array([0,3.5,10,25,35,50,110])
#    phi_bin = np.linspace(-3.14, 3.14, 21)
#    theta_bin = np.linspace(-1.57, 1.57, 21)

#    nd = np.size(data)

#    R3d = np.sqrt(data['x_gc']**2 + data['y_gc']**2 + data['z_gc']**2)
#    pos3d = np.array([R3d, data['phi'], data['theta'] ])
#    pos3d = np.reshape(pos3d, [3, nd]).T

#    hist, edges = np.histogramdd(pos3d, weights = np.ones(nd), bins=(rbin, phi_bin, theta_bin))
#    index_0 = np.where(hist >0)
#    space_cover_ratio = np.size(index_0) / np.size(hist) /3.0

#    print(np.size(hist), np.size(index_0))
#    print(space_cover_ratio)

#    data['ws'] = np.ones(np.size(data))
#    data_ag['ws']=np.ones(np.size(data_ag))

#    for io in range(0, norbit):
#        pos3d_io = np.array([data_ag['r3d'][io*tra: (io+1)*tra], data_ag['phi'][io*tra: (io+1)*tra], data_ag['theta'][io*tra: (io+1)*tra] ])
#        pos3d_io = np.reshape(pos3d_io, [3, tra]).T
#        hist_io, edges = np.histogramdd(pos3d_io, bins=(rbin, phi_bin, theta_bin))
#        data['ws'][io] = np.sum(hist_io)/np.sum(hist_io[index_0])
#        data_ag['ws'][io*tra : (io+1)*tra] = np.sum(hist_io)/np.sum(hist_io[index_0])


#    data_ag.write(base_path + model +r'/Orbits/orbits_'+s_file_affix + '.fits', overwrite = True)

#############################################
   # velocity distributions in 3D space
    nRz = 25
    Rzmax = 50 
    nphi = 4
    

    # density distribution from observations.
    dtfile = base_path + '/data_for_model/lamost_dr8_SFlast_cut4_4phi/w4_SB_Rz_254phi_err.txt'
    den_Rz, den_err = Read_obsSB_4phi(dtfile, nRz, nphi)

    # calculate density distribution from the model
#    data_ag['ws'][data_ag['ws']>5] = 5
    data_ag['w0'][data_ag['w0']>20] = 20

    z2d, R2d, den_m = calculate_RzSB_4phi(data_ag, nRz, Rzmax, nphi)
    den_m = den_m/(npart/2.)
    den_m[den_Rz==0] = 0
    

    # normalize the density distribution with the flux of the data
    r3d_bin = np.sqrt(R2d**2 + z2d**2)
    model_flux = np.sum(den_m[r3d_bin>10]* (2*np.pi/nphi * R2d[r3d_bin>10]))
    data_flux = np.sum(den_Rz[r3d_bin>10] * (2*np.pi/nphi * R2d[r3d_bin>10]))
    den_m = den_m * data_flux / model_flux

    
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

    plot_flag =1
    if plot_flag ==1:

        vmax = np.log10(np.max(den_Rz))
        vmin = -1 #np.log10(np.min(hist))


        font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 20}
        mpl.rc('font', **font)

        fig = plt.figure(figsize=(32, 20))

        for i in range(nphi):
            ax = fig.add_subplot(3, 5, i+1)
    
            hist_slice = np.log10(den_Rz[:,:,i])

            im = ax.imshow(hist_slice,  
                   origin='lower',
                   extent=[0, Rzmax, 0, Rzmax],
                   aspect='auto',
                   cmap='jet',
                   vmin=vmin,
                   vmax=vmax)
 #           plt.text(10,-50,'phi=%.2f'%phi_bin[i]+' - %.2f'%phi_bin[i+1])
            if i== nphi-1: 
                axins1 = inset_axes(ax,
                    width="10%",  # width = 50% of parent_bbox width
                    height="80%",  # height : 5%
                    loc='center right',
                    bbox_to_anchor=((0.8, 0, 0.2, 1.0)),
                    bbox_transform=ax.transAxes,
                    borderpad=0)
                cbar=fig.colorbar(im, cax = axins1, orientation='vertical', pad=0.1)
                cbar.ax.tick_params(labelsize='x-small')
                cbar.set_label(label=r'$\log(\rho_{\rm star}$ [N/kpc$^3$])', fontsize = 15)
            if i == 0:
                plt.xlabel(r'$R_{\rm gc}$ [kpc]')
                plt.ylabel(r'$Z_{\rm gc}$ [kpc]')

         ########## plot model #################
            for i in range(nphi):
                ax = fig.add_subplot(3, 5, 5+1+i)

                hist_slice = np.log10(den_m[:,:,i])
                im = ax.imshow(hist_slice,
                   origin='lower',
                   extent=[0, Rzmax, 0, Rzmax],
                   aspect='auto',
                   cmap='jet',
                   vmin=vmin,
                   vmax=vmax)
  #              plt.text(10,-50,'phi=%.2f'%phi_bin[i]+' - %.2f'%phi_bin[i+1])

            if i== nphi-1:
                axins1 = inset_axes(ax,
                    width="10%",  # width = 50% of parent_bbox width
                    height="80%",  # height : 5%
                    loc='center right',
                    bbox_to_anchor=((0.8, 0, 0.2, 1.0)),
                    bbox_transform=ax.transAxes,
                    borderpad=0)
                cbar=fig.colorbar(im, cax = axins1, orientation='vertical', pad=0.1)
                cbar.ax.tick_params(labelsize='x-small')
                cbar.set_label(label=r'$\log(\rho_{\rm model}$ [N/kpc$^3$])', fontsize = 15)
            if i == 0:
                plt.xlabel(r'$R_{\rm gc}$ [kpc]')
                plt.ylabel(r'$Z_{\rm gc}$ [kpc]')

        #################################################################
        ## plot the residual
        #################################################################

        vmax = 3
        vmin = -3

        for i in range(nphi):
            ax = fig.add_subplot(3, 5, 2*5+1+i)
    
            den_Rz[den_Rz==0] = float('nan')
            hist_slice = (den_Rz[:,:,i]- den_m[:,:,i])/den_err[:,:,i]

            im = ax.imshow(hist_slice,  
                   origin='lower',
                   extent=[0, Rzmax, 0, Rzmax],
                   aspect='auto',
                   cmap='jet',
                   vmin=vmin,
                   vmax=vmax)
            if i== nphi-1: 
                axins1 = inset_axes(ax,
                    width="10%",  # width = 50% of parent_bbox width
                    height="80%",  # height : 5%
                    loc='center right',
                    bbox_to_anchor=((0.8, 0, 0.2, 1.0)),
                    bbox_transform=ax.transAxes,
                    borderpad=0)
                cbar=fig.colorbar(im, cax = axins1, orientation='vertical', pad=0.1)
                cbar.ax.tick_params(labelsize='x-small')
                cbar.set_label(label=r'Residual', fontsize = 15)
    
            if i == 0:
                plt.xlabel(r'$R_{\rm gc}$ [kpc]')
                plt.ylabel(r'$Z_{\rm gc}$ [kpc]')
        plt.savefig(base_path + model+'/SB_Rz/'+ s_file_affix + '_SBRz.pdf')
        plt.close()


    #################################################
    # calculate the velocity histograms from model
    #################################################
#    nv = 201
#    vbd =  np.linspace(-800, 800, nv+1)

#    rbd = np.array([4, 6, 8, 10, 12, 15, 20, 30, 50])
#    nr = int(np.size(rbd)-1)

#    tbd = np.array([0,15,30,45,60,90])
#    ntheta = int(np.size(tbd)-1)

#    pbd = np.linspace(-180, 180, nphi+1)

#    vr_mw, vp_mw, vt_mw = calculate_vhist_weight_4phi(data_ag, rbd, tbd, pbd, vbd)

    #t= table.Table()
    #t['vr'] = vr_mw.flatten()
    #t['vp'] = vp_mw.flatten()
    #t['vt'] = vt_mw.flatten()
    #t.write(base_path + model + r'/vvhist/vv_hist_'+s_file_affix+'.fits', format = 'ascii',overwrite=True)

    # calculate LL in 1-nr bins
#    nr_end = nr
#    nr_s = 2
#    ntheta_s = 0

#    ll_vr_w = loglike_cal(data['r_gc'], data['theta'], data['phi'], data['vr_gc'], data['vr_err'], vr_mw, nr_end, nr_s, ntheta,ntheta_s, nv, rbd, tbd, pbd, vbd)

#    ll_vp_w = loglike_cal(data['r_gc'], data['theta'], data['phi'], data['v_phi'], data['vphi_err'], vp_mw, nr_end,nr_s, ntheta,ntheta_s, nv, rbd, tbd, pbd, vbd)

#    ll_vt_w = loglike_cal(data['r_gc'], data['theta'], data['phi'], data['v_the'], data['vthe_err'],vt_mw, nr_end,nr_s, ntheta,ntheta_s, nv, rbd, tbd, pbd, vbd)


    # Calculate the likelihood of data from South
#    dtfile_s = base_path + '/data_for_model/lamost_dr8_SFlast_cut4_4phi/halo_clean_S.txt'

#    data_s = table.Table.read(dtfile_s, format='ascii')
#    ll_vr = loglike_cal(data_s['r_gc'], data_s['theta'], data_s['phi'], data_s['vr_gc'], data_s['vr_err'], vr_mw, nr_end, nr_s, ntheta,ntheta_s, nv, rbd, tbd, pbd, vbd)
    
#    ll_vp = loglike_cal(data_s['r_gc'], data_s['theta'], data_s['phi'], data_s['v_phi'], data_s['vphi_err'], vp_mw, nr_end,nr_s, ntheta,ntheta_s, nv, rbd, tbd, pbd, vbd)

#    ll_vt = loglike_cal(data_s['r_gc'], data_s['theta'], data_s['phi'], data_s['v_the'], data_s['vthe_err'],vt_mw, nr_end,nr_s, ntheta,ntheta_s, nv, rbd, tbd, pbd, vbd)


#    ppfile = base_path + model +'/llp/'+s_file_affix +'.txt'
#    df = open(ppfile,'w')
#    df.write(("%6.3f" % rho0) + '  '+("%4.3f" % rs) + '  '+ ("%5.3f" % phalo) + '  '+ ("%5.3f" % qhalo)+  '  '+ ("%5.1f" % alpha_halo) +  '  '+ ("%5.1f" % beta_halo)  +  '  '+ ("%5.3f" % gamma) + '  ' + ("%8.5e" % ll_vr) + '  ' + ("%8.5e" % ll_vp) + '  ' + ("%8.5e" % ll_vt) + '  ' + ("%8.5e" % ll_vr_w) + '  ' + ("%8.5e" % ll_vp_w) + '  ' + ("%8.5e" % ll_vt_w) + '  ' + ("%8.5e" % chi2_SB) )
    

#    ll_tot = ( ll_vr +  ll_vp +  ll_vt + ll_vr_w + ll_vp_w +  ll_vt_w) - 0.5* chi2_SB
   


    del data_ag, ag_out, data #, data_s
    ll_tot = - 0.5* chi2_SB
    return ll_tot
    #-----------------------------------------------------
    

