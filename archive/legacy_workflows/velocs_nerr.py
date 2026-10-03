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
#########################xy to Rphi##################################
def xy_Rphi_nerr(x, y, mc=None, n=100):
  t = ast.Table()
  t['R']   = (x**2 + y**2)**0.5
  t['phi'] = np.arctan2(y,x)%(2*np.pi); t['phi'].unit = u.rad
  

  return t['R'], t['phi']




#########################vxyz_gc to vlbr_gc#######################################
def vxyz_gc_vrtp_gc(x, y, z, vx,  vy,  vz):
  t = ast.Table()
  
  r    = np.sqrt(x**2 + y**2 + z**2)
  phi = np.arctan2(y,x) #phi
  theta = np.arcsin(z/r)  # theta
  
  t['v_r'] =   vx * np.cos(phi) * np.cos(theta) + vy * np.sin(phi) * np.cos(theta) + vz * np.sin(theta); t['v_r'].unit = u.km/u.s
  t['v_phi'] = - vx * np.sin(phi) + vy * np.cos(phi); t['v_phi'].unit = u.km/u.s
  t['v_the'] = - vx * np.cos(phi) * np.sin(theta) - vy * np.sin(phi) * np.sin(theta) + vz * np.cos(theta); t['v_the'].unit = u.km/u.s
  
  
  t['r'] = r
  t['phi'] = phi
  t['theta'] = theta

  return  t['v_r'], t['v_phi'], t['v_the'], t['r'], t['phi'], t['theta']


######################### vlbr_gc to vxyz_gc #######################################
def vrtp_gc_vxyz_gc(r, phi, theta, v_r,  v_phi,  v_the):
  t = ast.Table()
  
  z = r* np.sin(theta)
  R2d = r*np.cos(theta)
  x = R2d * np.cos(phi)
  y = R2d * np.sin(phi)
  
  vz   = v_r * np.sin(theta) + v_the * np.cos(theta)
  VR2d = v_r * np.cos(theta) - v_the * np.sin(theta)
  vy = VR2d * np.sin(phi) + v_phi * np.cos(phi)
  vx = VR2d * np.cos(phi) - v_phi * np.sin(phi)
  
  
  t['vx_gc'] =   vx; t['vx_gc'].unit = u.km/u.s
  t['vy_gc'] =   vy; t['vy_gc'].unit = u.km/u.s
  t['vz_gc'] =   vz; t['vz_gc'].unit = u.km/u.s
  
  
  t['x_gc'] = x ; t['x_gc'].unit = u.kpc
  t['y_gc'] = y ; t['y_gc'].unit = u.kpc
  t['z_gc'] = z ; t['z_gc'].unit = u.kpc

  return  t['vx_gc'], t['vy_gc'], t['vz_gc'], t['x_gc'], t['y_gc'], t['z_gc']

