import astropy.table as ast
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.style.use('seaborn-bright')

from astropy import units as u
from astropy.coordinates import SkyCoord
from random import gauss
from scipy import stats 
from tqdm import tqdm

#########################v_sun & v_lsr######################################
#v_c = [0., 220., 0.] # Kerr (1986)
#v_c = [0., 225., 0.] # De Grijs (2017)

#v_sun = v_c + [10.,  5.2,   7.2] # Dehnen (1998)
#v_sun = v_c + [9.,   12.,   7.]  # Carollo (2010)
#v_sun = v_c + [11.1, 12.24, 7.25] # Schonorich (2014)

v_sun = [11.1, 225. + 12.24, 7.25]

#########################hrv to v_los#######################################
def hrv_vlos(l, b, hrv, hrv_err, v_sun = v_sun):
  v_los = hrv + v_sun[0] * np.cos(np.deg2rad(l)) * np.cos(np.deg2rad(b))\
            	+ v_sun[1] * np.sin(np.deg2rad(l)) * np.cos(np.deg2rad(b))\
            	+ v_sun[2] * np.sin(np.deg2rad(b))

  v_los_err = hrv_err

  return v_los*u.km/u.s, v_los_err*u.km/u.s

def vlos_hrv(l, b, vlos, vlos_err, v_sun = v_sun):

  hrv =  vlos - v_sun[0] * np.cos(np.deg2rad(l)) * np.cos(np.deg2rad(b))\
              - v_sun[1] * np.sin(np.deg2rad(l)) * np.cos(np.deg2rad(b))\
              - v_sun[2] * np.sin(np.deg2rad(b))

  hrv_err = vlos_err

  return hrv*u.km/u.s, hrv_err*u.km/u.s

#########################pm to vlb_sun & vlb_gc#######################################
def pm_vlb_sun(ra, dec, l, b, pmra, pmdec, pmra_err, pmdec_err, d, d_err, v_sun=v_sun, mc=None, n=100):
  ##########Radoslaw Poleski_<Transformation of the equatorial proper motion to the Galactic system>###########
  alpha_G = 192.85948
  delta_G = 27.12825
  
  C1 = np.sin(np.deg2rad(delta_G)) * np.cos(np.deg2rad(dec)) - \
       np.cos(np.deg2rad(delta_G)) * np.sin(np.deg2rad(dec)) * \
       np.cos(np.deg2rad(ra)       - np.deg2rad(alpha_G))

  C2 = np.cos(np.deg2rad(delta_G)) * np.sin(np.deg2rad(ra - alpha_G))

  cos_b = np.sqrt(C1**2 + C2**2)

  pl = 1/cos_b*(C1*np.array(pmra) + C2*np.array(pmdec))
  pb = 1/cos_b*(C1*np.array(pmdec) - C2*np.array(pmra))

  pl_err = np.sqrt((C1*np.array(pmra_err)/cos_b)**2 + (C2*np.array(pmdec_err))**2)
  pb_err = np.sqrt((C1*np.array(pmdec_err)/cos_b)**2 + (C2*np.array(pmra_err))**2)
  #########Galactic proper motion to Galactic tangential velocities###########
  t = ast.Table()
  t['pl'] = pl; t['pb'] = pb
  t['vl_sun'] = 4.74 * pl * d; t['vl_sun'].unit = u.km/u.s
  t['vb_sun'] = 4.74 * pb * d; t['vb_sun'].unit = u.km/u.s
  
  t['vl_err'] = np.sqrt((4.74*(d*pl_err))**2 + (4.74*(pl*d_err))**2); t['vl_err'].unit = u.km/u.s
  t['vb_err'] = np.sqrt((4.74*(d*pb_err))**2 + (4.74*(pb*d_err))**2); t['vb_err'].unit = u.km/u.s

  #########vlb_gsr########### 
  t['vl_gsr'] = t['vl_sun'] - v_sun[0] * np.sin(np.deg2rad(l)) + v_sun[1] * np.cos(np.deg2rad(l)); t['vl_gsr'].unit = u.km/u.s
  t['vb_gsr'] = t['vb_sun'] - v_sun[0] * np.cos(np.deg2rad(l)) * np.sin(np.deg2rad(b))\
                            - v_sun[1] * np.sin(np.deg2rad(l)) * np.sin(np.deg2rad(b))\
                            + v_sun[2] * np.cos(np.deg2rad(b)); t['vb_gsr'].unit = u.km/u.s

  #########Monte Carlo errors########### 
  if mc:
    gs_t = ast.Table()
    for names in ['vl', 'vb']:
      gs_t[names + '_sun_gs'] = np.zeros(n)
      gs_t[names + '_sun_gs'] = np.zeros(n)
    
    mc_t = ast.Table()
    for names in ['vl', 'vb']:
      mc_t[names + '_sun_mc'] = np.zeros(len(ra))
      mc_t[names + '_err_mc'] = np.zeros(len(ra))
    
    for i in tqdm(range(len(ra)), ascii = 1, ncols = 100):
      for j in range(n):
        pmra_gs  = gauss(pmra[i],  pmra_err[i])
        pmdec_gs = gauss(pmdec[i], pmdec_err[i])
        d_gs     = gauss(d[i], d_err[i])
    
        C1 = np.sin(np.deg2rad(delta_G)) * np.cos(np.deg2rad(dec[i])) - \
           np.cos(np.deg2rad(delta_G)) * np.sin(np.deg2rad(dec[i])) * \
           np.cos(np.deg2rad(ra[i]) - np.deg2rad(alpha_G))
        
        C2 = np.cos(np.deg2rad(delta_G)) * np.sin(np.deg2rad(ra[i] - alpha_G))
        
        cos_b = np.sqrt(C1**2 + C2**2)
    
        pl = 1/cos_b*(C1*np.array(pmra_gs) + C2*np.array(pmdec_gs))
        pb = 1/cos_b*(C1*np.array(pmdec_gs) - C2*np.array(pmra_gs))
    
        #########Galactic proper motion to Galactic tangential velocities#############
        gs_t['vl_sun_gs'][j] = 4.74 * pl * d_gs
        gs_t['vb_sun_gs'][j] = 4.74 * pb * d_gs
    
      mc_t['vl_sun_mc'][i], mc_t['vl_err_mc'][i] = stats.norm.fit(gs_t['vl_sun_gs'])
      mc_t['vb_sun_mc'][i], mc_t['vb_err_mc'][i] = stats.norm.fit(gs_t['vb_sun_gs'])
    
    for names in ['vl_sun', 'vl_err', 'vb_sun', 'vb_err']:
        fig = plt.figure(figsize = ((8,6)))
        ax = [0.18, 0.18, 0.70, 0.8]
        ax = plt.axes(ax)
    
        ax.scatter(t[names], mc_t[names + '_mc'], c = 'r')
    
        plt.xlabel(names + ' (km s$^{-1}$)', fontsize = 25)
        plt.ylabel(names + '_mc' + ' (km s$^{-1}$)',  fontsize = 25)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.show()
  
  return t['pl'], t['pb'], t['vl_sun'], t['vb_sun'], t['vl_gsr'], t['vb_gsr'], t['vl_err'], t['vb_err']

#########################vlb_sun to vxyz#######################################
def vlb_sun_vxyz(l, b, vl_sun, vb_sun, hrv, vl_err, vb_err, hrv_err, v_sun=v_sun, mc=None, n=100):
  t = ast.Table()
  t['vx_sun'] = - vl_sun * np.sin(np.deg2rad(l))\
              - vb_sun * np.sin(np.deg2rad(b)) * np.cos(np.deg2rad(l))\
              + hrv    * np.cos(np.deg2rad(b)) * np.cos(np.deg2rad(l)); t['vx_sun'].unit = u.km/u.s
                 
  t['vy_sun'] = vl_sun * np.cos(np.deg2rad(l))\
              - vb_sun * np.sin(np.deg2rad(b)) * np.sin(np.deg2rad(l))\
              + hrv    * np.cos(np.deg2rad(b)) * np.sin(np.deg2rad(l)); t['vy_sun'].unit = u.km/u.s
                 
  t['vz_sun'] = vb_sun * np.cos(np.deg2rad(b)) + hrv * np.sin(np.deg2rad(b)); t['vz_sun'].unit = u.km/u.s
  
  t['vx_err'] = np.sqrt(
                          (np.sin(np.deg2rad(l)) * vl_err)**2\
                        + (np.sin(np.deg2rad(b)) * np.cos(np.deg2rad(l)) * vb_err)**2\
                        + (np.cos(np.deg2rad(b)) * np.cos(np.deg2rad(l)) * hrv_err)**2); t['vx_err'].unit = u.km/u.s
  
  t['vy_err'] = np.sqrt(
                          (np.cos(np.deg2rad(l)) * vl_err)**2\
                        + (np.sin(np.deg2rad(b)) * np.sin(np.deg2rad(l)) * vb_err)**2\
                        + (np.cos(np.deg2rad(b)) * np.sin(np.deg2rad(l)) * hrv_err)**2) ; t['vy_err'].unit = u.km/u.s
  
  t['vz_err'] = np.sqrt(
                          (np.cos(np.deg2rad(b)) * vb_err)**2\
                        + (np.sin(np.deg2rad(b)) * hrv_err)**2); t['vz_err'].unit = u.km/u.s

  t['vx_gc'] = t['vx_sun'] + v_sun[0]; t['vx_gc'].unit = u.km/u.s 
  t['vy_gc'] = t['vy_sun'] + v_sun[1]; t['vy_gc'].unit = u.km/u.s 
  t['vz_gc'] = t['vz_sun'] + v_sun[2]; t['vz_gc'].unit = u.km/u.s 

  #########Monte Carlo errors########### 
  if mc:
    gs_t = ast.Table()
    for names in ['vx', 'vy', 'vz']:
      gs_t[names + '_sun_gs'] = np.zeros(n)
      gs_t[names + '_err_gs'] = np.zeros(n)
    
    mc_t = ast.Table()
    for names in ['vx', 'vy', 'vz']:
      mc_t[names + '_sun_mc'] = np.zeros(len(l))
      mc_t[names + '_err_mc'] = np.zeros(len(l))
    
    for i in tqdm(range(len(l)), ascii = 1, ncols = 100):
      for j in range(n):
        vl_sun_gs = gauss(vl_sun[i], vl_err[i])
        vb_sun_gs = gauss(vb_sun[i], vb_err[i])
        hrv_gs    = gauss(hrv   [i], hrv_err[i])
    
        gs_t['vx_sun_gs'][j] = - vl_sun_gs * np.sin(np.deg2rad(l[i]))\
                               - vb_sun_gs * np.sin(np.deg2rad(b[i])) * np.cos(np.deg2rad(l[i]))\
                               + hrv_gs    * np.cos(np.deg2rad(b[i])) * np.cos(np.deg2rad(l[i]))
    
        gs_t['vy_sun_gs'][j] =   vl_sun_gs * np.cos(np.deg2rad(l[i]))\
                               - vb_sun_gs * np.sin(np.deg2rad(b[i])) * np.sin(np.deg2rad(l[i]))\
                               + hrv_gs    * np.cos(np.deg2rad(b[i])) * np.sin(np.deg2rad(l[i]))
    
        gs_t['vz_sun_gs'][j] =   vb_sun_gs * np.cos(np.deg2rad(b[i])) + hrv_gs * np.sin(np.deg2rad(b[i]))
    
      mc_t['vx_sun_mc'][i], mc_t['vx_err_mc'][i] = stats.norm.fit(gs_t['vx_sun_gs'])
      mc_t['vy_sun_mc'][i], mc_t['vy_err_mc'][i] = stats.norm.fit(gs_t['vy_sun_gs'])
      mc_t['vz_sun_mc'][i], mc_t['vz_err_mc'][i] = stats.norm.fit(gs_t['vz_sun_gs'])
    
    for names in ['vx_sun', 'vx_err', 'vy_sun', 'vy_err', 'vz_sun', 'vz_err']:
      fig = plt.figure(figsize = ((8,6)))
      ax = [0.18, 0.18, 0.70, 0.8]
      ax = plt.axes(ax)
    
      ax.scatter(t[names], mc_t[names + '_mc'], c = 'r')
    
      plt.xlabel(names + ' (km s$^{-1}$)', fontsize = 25)
      plt.ylabel(names + '_mc' + ' (km s$^{-1}$)',  fontsize = 25)
      plt.xticks(fontsize = 15)
      plt.yticks(fontsize = 15)
      plt.show()

  return t['vx_sun'], t['vy_sun'], t['vz_sun'], t['vx_gc'], t['vy_gc'], t['vz_gc'], t['vx_err'], t['vy_err'], t['vz_err']


#########################vxyz_gc to vlbr_gc#######################################
def vxyz_gc_vlbr_gc(x, x_err, y, y_err, z, z_err, vx, vx_err, vy, vy_err, vz, vz_err, mc=None, n=100):
  t = ast.Table()
  
  r    = np.sqrt(x**2 + y**2 + z**2)
  gc_l = np.arctan2(y,x)
  gc_b = np.arcsin(z/r)
  
  r_err    = (x**2 + y**2 + z**2)**(-0.5) * ((x*x_err)**2 + (y*y_err)**2 + (z*z_err)**2)**0.5
  gc_l_err = ((x*y_err)**2 + (y*x_err)**2)**0.5 * (x**2 + y**2)**(-1)
  gc_b_err = (((r*z_err)**2 + (z*r_err)**2)**0.5) * ((r**2 * (r**2 - z**2))**(-0.5))
  
  t['vr_gc'] =   vx * np.cos(gc_l) * np.cos(gc_b) + vy * np.sin(gc_l) * np.cos(gc_b) + vz * np.sin(gc_b); t['vr_gc'].unit = u.km/u.s
  t['v_phi'] = - vx * np.sin(gc_l) + vy * np.cos(gc_l); t['v_phi'].unit = u.km/u.s
  t['v_the'] = - vx * np.cos(gc_l) * np.sin(gc_b) - vy * np.sin(gc_l) * np.sin(gc_b) + vz * np.cos(gc_b); t['v_the'].unit = u.km/u.s
  
  t['vr_gc_err'] = (( np.cos(gc_l)*np.cos(gc_b) * vx_err)**2 + (np.sin(gc_l)*np.sin(gc_b) * vy_err)**2 + (np.sin(gc_b) * vz_err)**2 + \
                   ((-np.sin(gc_l)*np.cos(gc_b)*vx + np.cos(gc_l)*np.cos(gc_b)*vy) * gc_l_err)**2 + \
                   ((-np.cos(gc_l)*np.sin(gc_b)*vx - np.sin(gc_l)*np.sin(gc_b)*vy + np.cos(gc_b)*vz) * gc_b_err)**2)**0.5; t['vr_gc_err'].unit = u.km/u.s 
  t['v_the_err'] = (( np.cos(gc_l)*np.sin(gc_b) * vx_err)**2 + (np.sin(gc_l)*np.sin(gc_b) * vy_err)**2 + (np.cos(gc_b) * vz_err)**2 + \
                   (( np.sin(gc_l)*np.sin(gc_b)*vx - np.sin(gc_b)*np.cos(gc_l)*vy) * gc_l_err)**2 + \
                   (( np.cos(gc_l)*np.cos(gc_b)*vx + np.sin(gc_l)*np.cos(gc_b)*vy + np.sin(gc_b)*vz) * gc_b_err)**2)**0.5; t['v_the_err'].unit = u.km/u.s 
  t['v_phi_err'] = (( np.sin(gc_l)*vx_err)**2 + (np.cos(gc_l)*vy_err)**2 + ((vx*np.cos(gc_l)+vy*np.sin(gc_l))*gc_l_err)**2)**0.5; t['v_phi_err'].unit = u.km/u.s 
  
  t['r'] = r; t['r_err'] = r_err; t['gc_l'] = gc_l; t['gc_l_err'] = gc_l_err; t['gc_b'] = gc_b; t['gc_b_err'] = gc_b_err

  #########Monte Carlo errors########### 
  if mc:
    gs_t = ast.Table()
    for names in ['vr_gc', 'v_phi', 'v_the', 'r', 'gc_l', 'gc_b']:
      gs_t[names + '_gs'] = np.zeros(n)
  
    mc_t = ast.Table()
    for names in ['vr_gc', 'v_phi', 'v_the', 'r', 'gc_l', 'gc_b']:
      mc_t[names + '_mc']     = np.zeros(len(x))
      mc_t[names + '_err_mc'] = np.zeros(len(x))
  
    for i in tqdm(range(len(x)), ascii = 1, ncols = 100):
      for j in range(n):
        x_gs = gauss(x[i], x_err[i])
        y_gs = gauss(y[i], y_err[i])
        z_gs = gauss(z[i], z_err[i])
    
        vx_gs = gauss(vx[i], vx_err[i])
        vy_gs = gauss(vy[i], vy_err[i])
        vz_gs = gauss(vz[i], vz_err[i])
    
        gs_t['r_gs'][j]    = np.sqrt(x_gs**2 + y_gs**2 + z_gs**2)
        gs_t['gc_l_gs'][j] = np.arctan2(y_gs, x_gs)
        gs_t['gc_b_gs'][j] = np.arcsin (z_gs/gs_t['r_gs'][j])  
  
        gs_t['vr_gc_gs'][j] =   vx_gs * np.cos(gs_t['gc_l_gs'][j]) * np.cos(gs_t['gc_b_gs'][j]) + vy_gs * np.sin(gs_t['gc_l_gs'][j]) \
                                      * np.cos(gs_t['gc_b_gs'][j]) + vz_gs * np.sin(gs_t['gc_b_gs'][j])
        gs_t['v_phi_gs'][j] = - vx_gs * np.sin(gs_t['gc_l_gs'][j]) + vy_gs * np.cos(gs_t['gc_l_gs'][j])
        gs_t['v_the_gs'][j] = - vx_gs * np.cos(gs_t['gc_l_gs'][j]) * np.sin(gs_t['gc_b_gs'][j]) - vy_gs * np.sin(gs_t['gc_l_gs'][j]) \
                                      * np.sin(gs_t['gc_b_gs'][j]) + vz_gs * np.cos(gs_t['gc_b_gs'][j])
    
      mc_t['r_mc'][i],    mc_t['r_err_mc'][i]    = stats.norm.fit(gs_t['r_gs'])
      mc_t['gc_l_mc'][i], mc_t['gc_l_err_mc'][i] = stats.norm.fit(gs_t['gc_l_gs'])
      mc_t['gc_b_mc'][i], mc_t['gc_b_err_mc'][i] = stats.norm.fit(gs_t['gc_b_gs'])
    
      mc_t['vr_gc_mc'][i], mc_t['vr_gc_err_mc'][i] = stats.norm.fit(gs_t['vr_gc_gs'])
      mc_t['v_phi_mc'][i], mc_t['v_phi_err_mc'][i] = stats.norm.fit(gs_t['v_phi_gs'])
      mc_t['v_the_mc'][i], mc_t['v_the_err_mc'][i] = stats.norm.fit(gs_t['v_the_gs'])
  
    for names in ['r', 'r_err', 'gc_l', 'gc_l_err', 'gc_b', 'gc_b_err', 'vr_gc', 'vr_gc_err', 'v_phi', 'v_phi_err', 'v_the', 'v_the_err']:
        fig = plt.figure(figsize = ((8,6)))
        ax = [0.18, 0.18, 0.70, 0.8]
        ax = plt.axes(ax)
    
        ax.scatter(t[names], mc_t[names + '_mc'], c = 'r')
    
        plt.xlabel(names + ' (km s$^{-1}$)', fontsize = 25)
        plt.ylabel(names + '_mc' + ' (km s$^{-1}$)',  fontsize = 25)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.show()

  return  t['vr_gc'], t['v_phi'], t['v_the'], t['vr_gc_err'], t['v_phi_err'], t['v_the_err']




#########################vxyz_gc to vlbr_gc#######################################
def vxy_gc_vR_gc(x, x_err, y, y_err, vx, vx_err, vy, vy_err, mc=None, n=100):
  gc_l = np.arctan2(y,x)
  gc_l_err = ((x*y_err)**2 + (y*x_err)**2)**0.5 * (x**2 + y**2)**(-1)
  
  vR = vx * np.cos(gc_l) + vy * np.sin(gc_l)
  vR_err = ((np.cos(gc_l)*vx_err)**2 + (np.sin(gc_l)*vy_err)**2 + ((-vx*np.sin(gc_l) + vy*np.cos(gc_l))*gc_l_err)**2)**0.5
  
  t = ast.Table()
  t['vR'] = vR; t['vR'].unit =  u.km/u.s
  t['vR_err'] = vR_err; t['vR_err'].unit =  u.km/u.s
  
  #########Monte Carlo errors########### 
  if mc:
    gs_t = ast.Table()
    gs_t['vR_gs'] = np.zeros(n)
    
    mc_t = ast.Table()
    mc_t['vR_mc']     = np.zeros(len(x))
    mc_t['vR_err_mc'] = np.zeros(len(x))
    
    for i in tqdm(range(len(x)), ascii = 1, ncols = 100):
      for j in range(n):
        x_gs  = gauss(x[i],  x_err[i]) 
        y_gs  = gauss(y[i],  y_err[i]) 
        vx_gs = gauss(vx[i], vx_err[i]) 
        vy_gs = gauss(vy[i], vy_err[i]) 
    
        gc_l_gs = np.arctan2(y_gs, x_gs)
        gs_t['vR_gs'][j] = vx_gs * np.cos(gc_l_gs) + vy_gs * np.sin(gc_l_gs)
    
      mc_t['vR_mc'][i], mc_t['vR_err_mc'][i] = stats.norm.fit(gs_t['vR_gs'])
    
    for names in ['vR', 'vR_err']:
        fig = plt.figure(figsize = ((8,6)))
        ax = [0.18, 0.18, 0.70, 0.8]
        ax = plt.axes(ax)
    
        ax.scatter(t[names], mc_t[names + '_mc'], c = 'r')
    
        plt.xlabel(names + ' (km s$^{-1}$)', fontsize = 25)
        plt.ylabel(names + '_mc' + ' (km s$^{-1}$)',  fontsize = 25)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.show()

  return t['vR'], t['vR_err']


######################### vsph_gc to vxyz_gc #######################################
def vsph_gc_vxyz_gc(x, y, z, vr, vphi, vthe):

    r    = np.sqrt(x**2 + y**2 + z**2)
    gc_l = np.arctan2(y,x)
    gc_b = np.arcsin(z/r)

    t = ast.Table()
    t['vx'] = vr * np.cos(gc_b) * np.cos(gc_l) - vphi * np.sin(gc_l) - vthe * np.sin(gc_b) * np.cos(gc_l)
    t['vy'] = vr * np.cos(gc_b) * np.sin(gc_l) + vphi * np.cos(gc_l) - vthe * np.sin(gc_b) * np.sin(gc_l)
    t['vz'] = vr * np.sin(gc_b) + vthe * np.cos(gc_b)

    return t['vx'], t['vy'], t['vz']




######################### vsph_gc to vxyz_gc #######################################
def vcyl_gc_vxy_gc(R, phi, vR, vphi):
    
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    
    vx = vR * np.cos(phi) - vphi * np.sin(phi)
    vy = vR * np.sin(phi) + vphi * np.cos(phi)

    return x, y, vx, vy


######################### vsph_gc to vxyz_gc #######################################
def vxy_gc_vcyl_gc(x, y, vx, vy):
    
    R   = (x**2 + y**2)**0.5
    phi = np.arctan2(y,x)%(2*np.pi)

    vR   = vx * np.cos(phi) + vy * np.sin(phi)
    vphi = - vx * np.sin(phi) + vy * np.cos(phi)

    return R, phi, vR, vphi
  
