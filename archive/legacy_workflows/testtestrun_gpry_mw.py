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
import emcee
from time import time
from emcee.mpi_pool import MPIPool

#from mpipool import MPIExecutor
#from mpi4py.MPI import COMM_WORLD

from Bayes_oint_mw_disk2 import int_one_model
from numpy.random import rand


import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.datasets import load_boston

def ll_gp_one(prun, alpha_halo, beta_halo, train_data):

    rho0, rhors2, phalo, qhalo, gamma = prun
   
    rho0 = round( rho0, 3)
    rs = round((rhors2 - rho0)/2, 3)
    phalo = round(phalo, 3)
    qhalo = round(qhalo, 3)
    alpha_halo = round(alpha_halo, 3)
    beta_halo = round(beta_halo, 3)
    gamma = round(gamma, 3)


b = load_boston()

X = [pd.DataFrame(b['data'])]
y = b['target']
for i in range(50):
    X.append(pd.DataFrame(b['data']))
    y = np.append(y,b['target'])

X = pd.concat(X)
X = pd.concat([X,X[X.columns[0:8]]],axis=1)
print(X.values.shape,y.shape)

kernel = DotProduct() + WhiteKernel()
model_gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
model_gp.fit(X.values, y)


        
        
    return ll_tot


# initialise mpi pool
pool = MPIPool()

if not pool.is_master():
    # wait for instructions from the master process
    pool.wait()
    sys.exit(0)

'''
pool = MPIExecutor()
if pool.is_main():
  try:
    pool.map(len, ([],[]))
  finally:
    pool.shutdown()

# Wait for all the workers to finish and continue together
COMM_WORLD.Barrier()
print("All processes continue code execution")
'''

# start time
start = time()

base_path = '/home/lzhu/MW_Bayes/'
model = 'Model_emcmc_mw_200X300_a16'  

partfile ='data_for_model/lamost_dr8_SFlast_cut4_NS_LMCc/halo_clean_N.txt'
#Only use the data in North hemisphere to integrate the orbit and build the orbit-superposition model
#But calculate the Likelihood for stars in both the North and sorthern hemisphere

if not os.path.exists(base_path + model):
    os.mkdir(base_path + model)
    os.mkdir(base_path + model + '/Orbits')
    os.mkdir(base_path + model + '/SB_Rz')
    os.mkdir(base_path + model + '/params')
    os.mkdir(base_path + model + '/vvhist')
    os.mkdir(base_path + model + '/llp')


ndim, nwalkers, nruns = 5, 200, 300

rho0 = rand( nwalkers ) * 3 + 5.0
rs = rand(nwalkers) * 1 + 9.3   # sample log(rhos * rs^2)
phalo = rand(nwalkers) * 0.6 + 0.7
qhalo = rand(nwalkers) * 0.6 + 0.7
gamma = rand(nwalkers) * 1.2 + 0.4
alpha_halo = 0
beta_halo = 0


p0 = list( array( [rho0, rs, phalo, qhalo, gamma ] ).T )

# Set up the backend
# Don't forget to clear it in case the file already exists
filename =  base_path + model +  "/tutorial.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)
sampler = emcee.EnsembleSampler( nwalkers, ndim, ll_submit_one, pool=pool,moves = emcee.moves.StretchMove(a=1.6), args=[alpha_halo, beta_halo, base_path, model,partfile ] )

#new_backend = emcee.backends.HDFBackend(filename)
#print("Initial size: {0}".format(new_backend.iteration))
#sampler = emcee.EnsembleSampler( nwalkers, ndim, ll_submit_one, pool=pool,
#    args=[alpha_halo, beta_halo, base_path, model,partfile ], backend=new_backend )

# --------------------------------------

# run mcmc
fsave = base_path + model + '/sample.dat'
fout = base_path + model + '/out.txt'
ncopy =  1
count = 1

#for posn, lnp, state in sampler.sample( None, iterations=nruns ):  # continue from new_backend
for posn, lnp, state in sampler.sample( p0, iterations=nruns ):
    print("run {:}".format( count ))
        
    for k in range( posn.shape[0] ):
        
        strout = ("%6.3f" % posn[k][0]) + '  '+("%5.3f" % posn[k][1]) + '  '+ ("%5.3f" % posn[k][2]) + '  '+ ("%5.3f" % posn[k][3])  +  '  '+ ("%5.3f" % posn[k][4]) + '  '+ ("%10.5e" % lnp[k]) + '\n'

        f = open( fsave, "a" )

        f.write( strout )
        f.close()
    
    if count % ncopy == 0:
        
        maf = mean( sampler.acceptance_fraction )
        f = open( fout, "a" )
        f.write("%3.0f" % count + '\n')
        f.write("  mean acceptance fraction:" + "%5.2f" % maf + '\n')
        f.close()
    
    count += 1


# --------------------------------------

# --------------------------------------

print( "")
print( "final stats")

maf = mean( sampler.acceptance_fraction )
print( "  mean acceptance fraction:", maf)

# --------------------------------------


# close the processes
pool.close()

# time taken
t = time() - start
ts = t % 60.
tm = int( ( t - ts ) /60. )
print( "" )
print( "time: {:} minutes {:.3f} seconds".format( tm, ts ) )
