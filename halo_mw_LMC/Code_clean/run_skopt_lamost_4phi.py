#!/usr/bin/env python
# coding: utf-8
import numpy as np
from numpy import abs, array, ceil, floor, log10, mean, ones,zeros, inf
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import table
import os
import sys
import subprocess
#import emcee
from time import time
#from emcee.mpi_pool import MPIPool
#from gpry.run import Runner
#from mpipool import MPIExecutor
#from mpi4py.MPI import COMM_WORLD

#import mpi4py
#mpi4py.rc.initialize = False  # do not initialize MPI automatically
#mpi4py.rc.finalize = False    # do not finalize MPI automatically
#from mpi4py import MPI


from skopt_oint_lamost_4phi import int_one_model
from numpy.random import rand
import numpy as np
from skopt import gp_minimize
from skopt import Optimizer
from skopt.space import Real, Integer
from skopt.utils import use_named_args



def ll_submit_one(prun):

    base_path = '/home/tqiu/research/halo_mw_LMC/'
    model = 'model_skopt'

    partfile ='data_for_model/lamost_dr8_SFlast_cut4_4phi/halo_clean_N.txt'
    alpha_halo = 0 #prun[5]
    beta_halo =  0 #prun[6]

    qhalo = prun[0]
    phalo = prun[1]
    rho0 = prun[2]
    rhors2 = prun[3]
    gamma = prun[4]

    phalo = round(phalo, 3)
    qhalo = round(qhalo, 3)
   
    rho0 = round( rho0, 3)
    rs = round((rhors2 - rho0)/2, 3)
    gamma = round(gamma, 3)
    alpha_halo = round(alpha_halo, 3)
    beta_halo = round(beta_halo, 3)
    


    dtfile =base_path+ partfile

    ll_tot = -int_one_model(base_path,model, rho0, rs, phalo, qhalo, alpha_halo, beta_halo, gamma, dtfile)    
        
    return ll_tot


#(base_path,model, rho0, rs, phalo, qhalo, alpha_halo, beta_halo, gamma, dtfile):
# initialise mpi pool
#pool = MPIPool()

#if not pool.is_master():
    # wait for instructions from the master process
#    pool.wait()
#    sys.exit(0)


# start time
start = time()

base_path = '/home/tqiu/research/halo_mw_LMC/'
model = 'model_skopt'

#Only use the data in North hemisphere to integrate the orbit and build the orbit-superposition model
#But calculate the Likelihood for stars in both the North and sorthern hemisphere

if not os.path.exists(base_path + model):
    os.mkdir(base_path + model)
    os.mkdir(base_path + model + '/Orbits')
    os.mkdir(base_path + model + '/SB_Rz')
    os.mkdir(base_path + model + '/params')
    os.mkdir(base_path + model + '/vvhist')
    os.mkdir(base_path + model + '/llp')


#y = ll_submit_one( [0.92, 0.84, 6.77, 9.77, 1.0])


pspace = [
    Real(0.5, 1.5, name='qhalo'),
    Real(0.1, 1.5, name='phalo'),
    Real(5, 8, name='rho0'),
    Real(9.3, 10.3, name='rs2rho'),
    Real(0.1, 3, name='gamma')
    #Real(0, 180, name='alpha_halo'),
    #Real(0, 90, name='beta_halo')
    ]
opt = Optimizer(pspace)

fsave = base_path + model + '/sample.dat'

for i in range(1000):
    suggested = opt.ask()
    y = ll_submit_one( suggested)
    opt.tell(suggested, y)
    print('iteration:', i, suggested, y)


    strout = ("%5.0f" % i) + '  '+("%5.3f" % suggested[0]) + '  '+ ("%5.3f" % suggested[1]) + '  '+ ("%5.3f" % suggested[2]) + '  '+ ("%5.3f" % suggested[3]) + '  '+ ("%5.3f" % suggested[4]) + '  '+ ("%10.5e" % y) + '\n'
    ff = open( fsave, "a" )
    ff.write( strout )
    ff.close()

# --------------------------------------

