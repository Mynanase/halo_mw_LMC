import astropy.table as ast
import numpy as np
import galpy.util.coords as gub

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.style.use('seaborn-bright')

from astropy import units as u
from random import gauss
from scipy import stats 
from tqdm import tqdm

#######################Angle Distance##########################
def d_a(l1, b1, l2, b2):
  l1 = np.deg2rad(l1)
  b1 = np.deg2rad(b1)
  l2 = np.deg2rad(l2)
  b2 = np.deg2rad(b2)

  d_a = np.arccos(np.sin(b1)*np.sin(b2) + np.cos(b1)*np.cos(b2)*np.cos(l1-l2))

  return np.rad2deg(d_a)

#################Distance modulus to Distance##################
def dm_d(dm, dm_err, mc=None, n=100):
  d     = 10**((dm + 5)/5.)
  d_err = (np.log(10.)*10**((dm+5)/5.)/5.)*dm_err

  t = ast.Table()
  t['d'] = d; t['d_err'] = d_err
  
  #########Monte Carlo errors########### 
  if mc:
    gs_t = ast.Table()
    gs_t['d_gs'] = np.zeros(n)
    
    mc_t = ast.Table()
    mc_t['d_mc']     = np.zeros(len(dm))
    mc_t['d_err_mc'] = np.zeros(len(dm))
    
    for i in tqdm(range(len(dm)), ascii = 1, ncols = 100):
      for j in range(n):
        dm_gs  = gauss(dm[i],  dm_err[i])
    
        gs_t['d_gs'][j] = 10**((dm_gs + 5)/5.)
    
      mc_t['d_mc'][i], mc_t['d_err_mc'][i] = stats.norm.fit(gs_t['d_gs'])
    
    for names in ['d', 'd_err']:
        fig = plt.figure(figsize = ((8,6)))
        ax = [0.18, 0.18, 0.70, 0.8]
        ax = plt.axes(ax)
    
        ax.scatter(t[names], mc_t[names + '_mc'], c = 'r')
    
        plt.xlabel(names + ' (pc)', fontsize = 25)
        plt.ylabel(names + '_mc' + ' (pc)',  fontsize = 25)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.show()

  return t['d']/1000., t['d_err']/1000.

#########################gal to xyz_sun#########################
def gal_xyz(l, b, d, d_err, mc=None, n=100):
    t = ast.Table()
    t['x'] = d * np.cos(np.deg2rad(b)) * np.cos(np.deg2rad(l)) * u.kpc
    t['y'] = d * np.cos(np.deg2rad(b)) * np.sin(np.deg2rad(l)) * u.kpc
    t['z'] = d * np.sin(np.deg2rad(b)) * u.kpc
    
    t['x_err'] = abs(d_err * np.cos(np.deg2rad(b)) * np.cos(np.deg2rad(l))) * u.kpc
    t['y_err'] = abs(d_err * np.cos(np.deg2rad(b)) * np.sin(np.deg2rad(l))) * u.kpc
    t['z_err'] = abs(d_err * np.sin(np.deg2rad(b))) * u.kpc
    
  #########Monte Carlo errors###########     
    if mc:
        gs_t = ast.Table()
        for names in ['x', 'y', 'z']:
            gs_t[names + '_gs'] = np.zeros(n)
    
        mc_t = ast.Table()
        for names in ['x', 'y', 'z']:
            mc_t[names + '_mc']     = np.zeros(len(l))
            mc_t[names + '_err_mc'] = np.zeros(len(l))
        
        for i in tqdm(range(len(l)), ascii = 1, ncols = 100):
          for j in range(n):
            d_gs = gauss(d[i], d_err[i])
        
            gs_t['x_gs'][j] = d_gs * np.cos(np.deg2rad(b[i])) * np.cos(np.deg2rad(l[i]))
            gs_t['y_gs'][j] = d_gs * np.cos(np.deg2rad(b[i])) * np.sin(np.deg2rad(l[i]))
            gs_t['z_gs'][j] = d_gs * np.sin(np.deg2rad(b[i]))
        
          mc_t['x_mc'][i], mc_t['x_err_mc'][i] = stats.norm.fit(gs_t['x_gs'])
          mc_t['y_mc'][i], mc_t['y_err_mc'][i] = stats.norm.fit(gs_t['y_gs'])
          mc_t['z_mc'][i], mc_t['z_err_mc'][i] = stats.norm.fit(gs_t['z_gs'])
    
        for names in ['x', 'x_err', 'y', 'y_err', 'z', 'z_err']:
            fig = plt.figure(figsize = ((8,6)))
            ax = [0.18, 0.18, 0.70, 0.8]
            ax = plt.axes(ax)
    
            ax.scatter(t[names], mc_t[names + '_mc'], c = 'r')
    
            plt.xlabel(names + ' (' + str(t[names].unit) + ')', fontsize = 25)
            plt.ylabel(names + '_mc' + ' (' + str(t[names].unit) + ')', fontsize = 25)
            plt.xticks(fontsize = 15)
            plt.yticks(fontsize = 15)
            plt.show()

    return t['x'], t['x_err'], t['y'], t['y_err'], t['z'], t['z_err']


def xyz_r(x, y, z, x_err, y_err, z_err, mc=None, n=100):
  t = ast.Table()
  t['r']   = (x**2 + y**2 + z**2)**0.5
  t['r_err'] = (((x*x_err)**2+(y*y_err)**2+(z*z_err)**2) / (x**2+y**2+z**2))**0.5

  #########Monte Carlo errors###########
  if mc:
    gs_t = ast.Table()
    for names in ['r']:
      gs_t[names + '_gs'] = np.zeros(n)
  
    mc_t = ast.Table()
    for names in ['r']:
      mc_t[names + '_mc']     = np.zeros(len(x))
      mc_t[names + '_err_mc'] = np.zeros(len(x))
    
    for i in tqdm(range(len(x)), ascii = 1, ncols = 100):
      for j in range(n):
        x_gs = gauss(x[i], x_err[i])
        y_gs = gauss(y[i], y_err[i])
        z_gs = gauss(z[i], z_err[i])
  
        gs_t['r_gs'][j] = (x_gs**2 + y_gs**2 + z_gs**2)**0.5
  
      mc_t['r_mc'][i],   mc_t['r_err_mc'][i] = stats.norm.fit(gs_t['r_gs'])
      
    for names in ['r', 'r_err']:
      fig = plt.figure(figsize = ((8,6)))
      ax = [0.18, 0.18, 0.70, 0.8]
      ax = plt.axes(ax)
      
      ax.scatter(t[names], mc_t[names + '_mc'], c = 'r')
      
      plt.xlabel(names + ' (' + str(t[names].unit) + ')', fontsize = 25)
      plt.ylabel(names + '_mc' + ' (' + str(t[names].unit) + ')', fontsize = 25)
      plt.xticks(fontsize = 15)
      plt.yticks(fontsize = 15)
      plt.show()

  return t['r'], t['r_err']

#########################Equ to Sgr#########################
def eq_sgr(ra, dec):
	ra  = np.deg2rad(ra)
	dec = np.deg2rad(dec)
	la2 = np.arctan2(-0.93595354 * np.cos(ra) * np.cos(dec) - 0.31910658 * np.sin(ra) * np.cos(dec) + 0.14886895 * np.sin(dec),
	                  0.21215555 * np.cos(ra) * np.cos(dec) - 0.84846291 * np.sin(ra) * np.cos(dec) - 0.48487186 * np.sin(dec))
	be2 =  np.arcsin( 0.28103559 * np.cos(ra) * np.cos(dec) - 0.42223415 * np.sin(ra) * np.cos(dec) + 0.86182209 * np.sin(dec))

	return np.rad2deg(la2)%360, np.rad2deg(be2)

#########################Sgr to Equ#########################
def sgr_eq(la2, be2):
  la2 = np.deg2rad(la2)
  be2 = np.deg2rad(be2)
  ra  = np.arctan2(- 0.84846291 * np.cos(la2) * np.cos(be2) - 0.31910658 * np.sin(la2) * np.cos(be2) - 0.42223415 * np.sin(be2),
                     0.21215555 * np.cos(la2) * np.cos(be2) - 0.93595354 * np.sin(la2) * np.cos(be2) + 0.28103559 * np.sin(be2))
  dec =  np.arcsin(- 0.48487186 * np.cos(la2) * np.cos(be2) + 0.14886895 * np.sin(la2) * np.cos(be2) + 0.86182209 * np.sin(be2))

  return np.rad2deg(ra), np.rad2deg(dec)

#########################xy to Rphi##################################
def xy_Rphi(x, y, x_err, y_err, mc=None, n=100):
  t = ast.Table()
  t['R']   = (x**2 + y**2)**0.5
  t['phi'] = np.arctan2(y,x)%(2*np.pi); t['phi'].unit = u.rad
  
  t['R_err'] = (x_err**2 + y_err**2)**0.5
  t['phi_err'] = ((x*y_err)**2 + (y*x_err)**2)**0.5 * (x**2 + y**2)**(-1)
  
  #########Monte Carlo errors###########
  if mc:
    gs_t = ast.Table()
    for names in ['R', 'phi']:
      gs_t[names + '_gs'] = np.zeros(n)

    mc_t = ast.Table()
    for names in ['R', 'phi']:
      mc_t[names + '_mc']     = np.zeros(len(x))
      mc_t[names + '_err_mc'] = np.zeros(len(x))
    
    for i in tqdm(range(len(x)), ascii = 1, ncols = 100):
      for j in range(n):
        x_gs = gauss(x[i], x_err[i])
        y_gs = gauss(y[i], y_err[i])
  
        gs_t['R_gs'][j]   = gub.rect_to_cyl(x_gs, y_gs, 0.)[0]
        gs_t['phi_gs'][j] = gub.rect_to_cyl(x_gs, y_gs, 0.)[1]
  
      mc_t['R_mc'][i],   mc_t['R_err_mc'][i] = stats.norm.fit(gs_t['R_gs'])
      mc_t['phi_mc'][i], mc_t['phi_err_mc'][i] = stats.norm.fit(gs_t['phi_gs'])
      
    for names in ['R', 'R_err', 'phi', 'phi_err']:
      fig = plt.figure(figsize = ((8,6)))
      ax = [0.18, 0.18, 0.70, 0.8]
      ax = plt.axes(ax)
      
      ax.scatter(t[names], mc_t[names + '_mc'], c = 'r')
      
      plt.xlabel(names + ' (' + str(t[names].unit) + ')', fontsize = 25)
      plt.ylabel(names + '_mc' + ' (' + str(t[names].unit) + ')', fontsize = 25)
      plt.xticks(fontsize = 15)
      plt.yticks(fontsize = 15)
      plt.show()

  return t['R'], t['R_err'], t['phi'], t['phi_err']