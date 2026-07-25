import agama as ag
import numpy as np
import os,sys,time
import galpy.util.conversion as conversion

from astropy import units as u
from galpy.orbit import Orbit
from galpy.util.coords import cyl_to_rect, galcencyl_to_vxvyvz

def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value

    return func_wrapper


@blockPrinting
def int_ag(ic, pot, t_p, tra, omg, per = False):


    #dt1 = time.time()
    if omg != False:
        orb = ag.orbit(ic = ic, potential = pot, time = t_p*pot.Tcirc(ic), trajsize = tra, Omega = omg)
    else:
        orb = ag.orbit(ic = ic, potential = pot, time = t_p*pot.Tcirc(ic), trajsize = tra)
    #dt2 = time.time()
    
   # print('Time to integrate orbit in Agama: %.4g s' % (dt2 - dt1))


    dt1 = time.time()
    orb_i = np.where(~np.isnan(pot.Tcirc(ic)))[0]
    orb_tl = np.tile(np.array(orb_i), (tra, 1)).T.flatten()

    t   = np.concatenate([orb[i][0] for i in orb_i])
    orb = np.concatenate([orb[i][1] for i in orb_i])

    ####### E #########
    x, y, z, vx, vy, vz = orb[:,0], orb[:,1], orb[:,2], orb[:,3], orb[:,4], orb[:,5]

    E_P = pot.potential(orb[:, :3]) # potential energy
    E_R = 0.5 * omg * (x*vy + y*vx) # rotation energy 
    E_K = 0.5 * (vx**2 + vy**2 + vz**2)**0.5 # kinematic energy
    E = E_P - E_R + E_K
    # E_mean = E.reshape(len(orb_i), tra).mean(axis = 1)



    if per == False:
        ag_out = {'orb_tl':orb_tl, 't':t, 'x': x, 'y': y, 'z': z,
         'vx': vx, 'vy': vy, 'vz': vz, 'E':E}
        #dt2 = time.time()
        #print('Table Concatenate Time: %.4g s' % (dt2 - dt1))
        
        return ag_out

    else:

        #dt1 = time.time()
        
        s_p = len(orb_i) * t_p # stars*periods of (stars*periods, trajsize)
        ######## z_max #########
        orb_z = z.reshape((s_p, len(orb)//s_p))
        z_max_orb = abs(orb_z).max(axis = 1) # z_max of each orbit
        z_max_star = z_max_orb.reshape(len(orb_i), t_p) # z_maxs of each star

        ######### apo ##########
        r = np.sum(orb[:,:3]**2, axis = 1)**0.5
        orb_r = r.reshape((int(s_p), len(orb)//s_p))

        apo_orb = orb_r.max(axis = 1) # apo of each orb
        apo_star = apo_orb.reshape(len(orb_i), t_p) # apo of each star

        ######### per ##########
        per_orb = orb_r.min(axis = 1)
        per_star = per_orb.reshape(len(orb_i), t_p)

        ######### ec ##########
        ec_star = (1. - per_star/apo_star)/(1. + per_star/apo_star)

        ag_out = {'orb_tl':orb_tl, 't':t, 'x': x, 'y': y, 'z': z, 
                  'vx': vx, 'vy': vy, 'vz': vz, 'E':E, 'z_max':z_max_star, 
                  'apo':apo_star, 'per':per_star, 'ec':ec_star}

        #dt2 = time.time()
        #print('Table Concatenate Time: %.4g s' % (dt2 - dt1))

        return ag_out






