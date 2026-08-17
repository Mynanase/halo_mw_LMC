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


from Bayes_oint_LMC_back import int_one_LMC
from numpy.random import rand
import numpy as np
from skopt import gp_minimize
from skopt import Optimizer
from skopt.space import Real, Integer
from skopt.utils import use_named_args



def ll_submit_one(prun):

    base_path = '/home/lzhu/halo_mw_LMC/'
    model = 'LMC_skopt'

    partfile ='data_for_model/lamost_dr8_SFlast_cut4_NS_LMCc/halo_clean_N.txt'
    alpha_halo = 0
    beta_halo = 0

    qhalo = prun[0]
    phalo = prun[1]
    rho0 = prun[2]
    rhors2 = prun[3]
    gamma = prun[4]

    massLMC = prun[5]

    phalo = round(phalo, 3)
    qhalo = round(qhalo, 3)
   
    rho0 = round( rho0, 3)
    rs = round((rhors2 - rho0)/2, 3)
    gamma = round(gamma, 3)
    alpha_halo = round(alpha_halo, 3)
    beta_halo = round(beta_halo, 3)
    


    dtfile =base_path+ partfile
   # if (rs < 0 or gamma <0 or gamma>2.5 or phalo<0. or qhalo <0 or rho0 <0):ll_tot = -inf
   # else:ll_tot = int_one_model(base_path,model, rho0, rs, phalo, qhalo, alpha_halo, beta_halo, gamma, dtfile)

    ll_tot = int_one_LMC(base_path,model, rho0, rs, phalo, qhalo, alpha_halo, beta_halo, gamma, massLMC, dtfile)
        
    return ll_tot



# initialise mpi pool
#pool = MPIPool()

#if not pool.is_master():
    # wait for instructions from the master process
#    pool.wait()
#    sys.exit(0)


# start time
start = time()

base_path = '/home/lzhu/halo_mw_LMC/'
model = 'LMC_skopt'

#Only use the data in North hemisphere to integrate the orbit and build the orbit-superposition model
#But calculate the Likelihood for stars in both the North and sorthern hemisphere

if not os.path.exists(base_path + model):
    os.mkdir(base_path + model)
    os.mkdir(base_path + model + '/figure')


pspace = [
    Real(0.5, 1.5, name='qhalo'),
    Real(0.5, 1.5, name='phalo'),
    Real(5, 8, name='rho0'),
    Real(9.3, 10.3, name='rs2rho'),
    Real(0.1, 1.8, name='gamma')
    Real(11.0, 11.3, name='massLMC')
    ]
opt = Optimizer(pspace)

fsave = base_path + model + '/sample.dat'

for i in range(500):
    suggested = opt.ask()
    y = ll_submit_one( suggested)
    opt.tell(suggested, y)
    print('iteration:', i, suggested, y)


    strout = ("%5.0f" % i) + '  '+("%5.3f" % suggested[0]) + '  '+ ("%5.3f" % suggested[1]) + '  '+ ("%5.3f" % suggested[2]) + '  '+ ("%5.3f" % suggested[3]) + '  '+ ("%5.3f" % suggested[4]) + '  '+ ("%5.3f" % suggested[5]) + '  '+ ("%10.5e" % y) + '\n'
    ff = open( fsave, "a" )
    ff.write( strout )
    ff.close()

# --------------------------------------

