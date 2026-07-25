#!/usr/bin/env python
# coding: utf-8
import numpy as np
import numpy
import scipy
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import table
import agama
from astropy import table, units, coordinates


def make_halo_density_rot(rho0, rs, alpha, beta, gamma, rcut, xi, phalo, qhalo, alpha_halo, beta_halo, gamma_halo):
    def dens_halo_rot(xyz):
        
        RR = np.zeros([3,3])
        RR[0, 0] = np.cos(alpha_halo)*np.cos(gamma_halo)-np.sin(alpha_halo)*np.cos(beta_halo)*np.sin(gamma_halo)
        RR[0, 1] = np.sin(alpha_halo)*np.cos(gamma_halo) + np.cos(alpha_halo)*np.cos(beta_halo)*np.sin(gamma_halo)
        RR[0, 2] = np.sin(beta_halo) * np.sin(gamma_halo)
        RR[1, 0] = -np.cos(alpha_halo)*np.sin(gamma_halo)-np.sin(alpha_halo)*np.cos(beta_halo)*np.cos(gamma_halo)
        RR[1, 1] = -np.sin(alpha_halo)*np.sin(gamma_halo) + np.cos(alpha_halo)*np.cos(beta_halo)*np.cos(gamma_halo)
        RR[1, 2] = np.sin(beta_halo) * np.cos(gamma_halo)
        RR[2, 0] = np.sin(alpha_halo) * np.sin(beta_halo)
        RR[2, 1] = -np.cos(alpha_halo) * np.sin(beta_halo)
        RR[2, 2] = np.cos(beta_halo)
        
        xyzh = xyz.copy() * 0.0
        xyzh.astype(np.float64)
        xyzh[:,0] = RR[0,0] * xyz[:,0] + RR[0,1] * xyz[:,1] + RR[0,2] * xyz[:,2]
        xyzh[:,1] = RR[1,0] * xyz[:,0] + RR[1,1] * xyz[:,1] + RR[1,2] * xyz[:,2]
        xyzh[:,2] = RR[2,0] * xyz[:,0] + RR[2,1] * xyz[:,1] + RR[2,2] * xyz[:,2]

        xh = xyzh[:,0]
        yh = xyzh[:,1]
        zh = xyzh[:,2]
        
        rellp =(phalo * qhalo)**(1/3.) * np.sqrt(xh**2 + (yh/phalo)**2 + (zh/qhalo)**2)
        #rellp = np.sqrt(xh**2 + (yh/phalo)**2 + (zh/qhalo)**2)
        rho_rot = rho0 * (rellp/rs)**(-gamma) * (1 + (rellp/rs)**alpha  )**((gamma-beta)/alpha) * np.exp(-(rellp/rcut)**xi)
        return rho_rot
    return dens_halo_rot
    
    

def int_one_LMC(base_path,model, rho0, rs, phalo, qhalo, alpha_halo, beta_halo, gamma, massLMC, dtfile):
    
#    print(rho0, rs, phalo, qhalo, alpha_halo, beta_halo, gamma)
    s_file_affix = 'rho0%4.3f_rs%4.3f_p%5.3f_q%5.3f_a%5.3f_b%5.3f_gamma%2.3f'%(rho0, rs, phalo, qhalo, alpha_halo, beta_halo, gamma)

    data = table.Table.read(dtfile, format='ascii')

    dt_ic= np.array([data[ 'x_gc'], data[ 'y_gc'], data[ 'z_gc'],
                  data['vx_gc'], data['vy_gc'], data['vz_gc']]).T
                  

    agama.setUnits(length=1, velocity=1, mass=1)  # work in units of 1 kpc, 1 km/s, 1 Msun)

    Trewind = -4.0  # initial time [Gyr] - the LMC orbit is computed back to that time
    Tcurr   =  0.0  # current time

    # heliocentric ICRS celestial coordinates and velocity of the LMC
    # (PM from Luri+ 2021, distance from Pietrzynski+ 2019, center and velocity from van der Marel+ 2002)
    ra, dec, dist, pmra, pmdec, vlos = 81.28, -69.78, 49.6, 1.858, 0.385, 262.2
    # transform to Galactocentric cartesian position/velocity, using built-in routines from Agama
    # (hence the manual conversion factors from degrees to radians and from mas/yr to km/s/kpc)
    l, b, pml, pmb = agama.transformCelestialCoords(agama.fromICRStoGalactic,ra * np.pi/180, dec * np.pi/180, pmra, pmdec)
    
    posvelLMC = agama.getGalactocentricFromGalactic(l, b, dist, pml*4.74, pmb*4.74, vlos)


    # define the MW Gravitational potential
    bar_mass = 10.2
    disk_mass =10.5
    
    pot_bulge = agama.Potential(type='Spheroid', mass = np.power(10,bar_mass) ,\
                            alpha = 1, gamma=0, beta=1.8, scaleRadius = 0.2, outerCutoffRadius=1.8, cutoffStrength=2)
    pot_THD = agama.Potential(type = 'MiyamotoNagai', mass = 6e9, scaleRadius = 2.0,
                          scaleHeight = 0.9) # thick disk
    pot_MD = agama.Potential(type = 'Disk', mass = np.power(10,disk_mass), scaleRadius = 2.6,
                          scaleHeight = 0.3, innerCutoffRadius = 7, sersicIndex=1) # thin disk

    pot_bulge2 = agama.Potential(type='Spheroid', mass = np.power(10,bar_mass) + np.power(10,disk_mass) ,\
                            alpha = 1, gamma=0, beta=1.8, scaleRadius = 0.2, outerCutoffRadius=1.8, cutoffStrength=2)

    
    alpha,beta,rcut,xi = 1,3,300,4

    gamma_halo = 0
    # alpha_halo + gamma_halo determines angle between x and X, we thus always set gamma_halo =0
    #rho0, rs, phalo, qhalo, alpha_halo, beta_halo, gamma

#    gamma = 1
#    rho0 = 6.5
#    rs = 20
#    phalo = 0.84
#    qhalo = 0.92



#    dens_my2  = make_halo_density_rot(np.power(10,rho0), rs, alpha, beta, gamma, rcut, xi, phalo, qhalo, alpha_halo*math.pi/180, beta_halo*math.pi/180, gamma_halo)

#    pot_halo = ag.Potential(type="Multipole", symmetry = 't', density=dens_my2, lmax=6, mmax=6, gridSizeR=36, rmin=1e-3, rmax=500 )


    pot_halo = agama.Potential(type='Spheroid', densityNorm = np.power(10, rho0), alpha = alpha, gamma=gamma, beta=beta, 
            scaleRadius = np.power(10, rs), p=phalo, q=qhalo, outerCutoffRadius=rcut, cutoffStrength=xi)


    densMWhalo = agama.Density(pot_halo)
    # For the calculation of LMC orbits, disk and bulge may affect the LMC orbits
    potMW = agama.Potential(pot_bulge, pot_THD, pot_MD, pot_halo)


    # create a sphericalized MW potential and a corresponding isotropic halo distribution function
    potMWsph   = agama.Potential(type='Multipole', potential=potMW, lmax=0, rmin=0.01, rmax=1000)
    gmHalo     = agama.GalaxyModel(potMWsph,
    agama.DistributionFunction(type='quasispherical', density=densMWhalo, potential=potMWsph))

    # compute the velocity dispersion in the MW halo needed for the dynamical friction
    rgrid      = np.logspace(1, 3, 16)
    xyzgrid    = np.column_stack([rgrid, rgrid*0, rgrid*0])
    sigmafnc   = agama.Spline(rgrid, gmHalo.moments(xyzgrid, dens=False, vel=False, vel2=True)[:,0]**0.5)

    # Create the LMC potential - a spherical truncated NFW profile with mass and radius
    # related by the equation below, which produces approximately the same enclosed mass
    # profile in the inner region, satisfying the observational constraints, as shown
    # in Fig.3 of Vasiliev,Belokurov&Erkal 2021.
    massLMC    = np.power(10, massLMC)  #1.5e11 #1.5e11
    radiusLMC  = (massLMC/1e11)**0.6 * 8.5
    bminCouLog = radiusLMC * 2.0   # minimum impact parameter in the Coulomb logarithm
    potLMC     = agama.Potential(
    type              = 'spheroid',
    mass              = massLMC,
    scaleradius       = radiusLMC,
    outercutoffradius = radiusLMC*10,
    gamma             = 1,
    beta              = 3)

######## PART ONE ########
# Simulate (approximately!) the past trajectory of the MW+LMC system under mutual gravity.
# Here, we integrate in time a 12-dimensional ODE system for positions & velocities of
# both galaxies in the external inertial reference frame. The acceleration of each galaxy
# is computed by taking the gradient of the rigid (non-deforming) potential of the other
# galaxy at the location of the first galaxy's center, and then assuming that the entire
# first galaxy experiences the same acceleration and continues to move as a rigid body.
# The same procedure then is applied in reverse. Moreover, we add a dynamical friction
# acceleration to the LMC, but not to the Milky Way; it is computed using the standard
# Chandrasekhar's formula, but with a spatially-varying value of Coulomb logarithm,
# which has been calibrated against full N-body simulations.
# This simplified model is certainly not physically correct, e.g. manifestly violates
# Newton's third law, but still captures the main features of the actual interaction.
    print("Computing the past orbits of the Milky Way and the LMC")

    def difeq(vars, t):
        x0    = vars[0:3]          # MW position
        v0    = vars[3:6]          # MW velocity
        x1    = vars[6:9]          # LMC position
        v1    = vars[9:12]         # LMC velocity
        dx    = x1-x0              # relative offset
        dv    = v1-v0              # relative velocity
        dist  = sum(dx**2)**0.5    # distance between the galaxies
        vmag  = sum(dv**2)**0.5    # magnitude of relative velocity
        f0    = potLMC.force(-dx)  # force from LMC acting on the MW center
        f1    = potMW .force( dx)  # force from MW acting on the LMC
        rho   = potMW.density(dx)  # actual MW density at this point
        sigma = sigmafnc(dist)     # approximate MW velocity dispersion at this point
        # distance-dependent Coulomb logarithm
        # (an approximation that best matches the results of N-body simulations)
        couLog= max(0, numpy.log(dist / bminCouLog)**0.5)
        X     = vmag / (sigma * 2**.5)
        drag  = -(4*numpy.pi * rho * dv / vmag *
            (scipy.special.erf(X) - 2/numpy.pi**.5 * X * numpy.exp(-X*X)) *
            massLMC * agama.G**2 / vmag**2 * couLog)   # dynamical friction force
        return numpy.hstack((v0, f0, v1, f1 + drag))

    Tstep   = 1./64
    tgrid   = numpy.linspace(Trewind, Tcurr, round((Tcurr-Trewind)/Tstep)+1)
    ic      = numpy.hstack((numpy.zeros(6), posvelLMC))
    sol     = scipy.integrate.odeint(difeq, ic, tgrid[::-1])[::-1]

# After obtaining the solution for trajectories of both galaxies,
# we transform it into a more convenient form, namely, into the non-inertial
# reference frame centered at the Milky Way center at all times.
# In this frame, the total time-dependent gravitational potential consists of
# three terms. First is the rigid potential of the Milky Way itself.
# Because the latter moves on a curvilinear trajectory, we need to add
# a corresponding spatially uniform acceleration field. Finally, the potential
# of the LMC is also rigid but moves in space.

# LMC trajectory in the MW-centric (non-inertial) reference frame
# (7 columns: time, 3 position and 3 velocity components)
    trajLMC = numpy.column_stack([tgrid, sol[:,6:12] - sol[:,0:6]])
# MW trajectory in the inertial frame
    trajMWx = agama.Spline(tgrid, sol[:,0], der=sol[:,3])
    trajMWy = agama.Spline(tgrid, sol[:,1], der=sol[:,4])
    trajMWz = agama.Spline(tgrid, sol[:,2], der=sol[:,5])
# MW centre acceleration is minus the second derivative of its trajectory in the inertial frame
    accMW   = numpy.column_stack([tgrid, -trajMWx(tgrid, 2), -trajMWy(tgrid, 2), -trajMWz(tgrid, 2)])
    potacc  = agama.Potential(type='UniformAcceleration', file=accMW)
    potLMCm = agama.Potential(potential=potLMC, center=trajLMC)  # potential of the moving LMC


# finally, the total time-dependent potential in the non-inertial MW-centric reference frame
# Use a spherical bulge and no disk to avoid the chaotic orbits in the inner regions
# Disk + bulge or triaxial bulge will cause a lot of chaotic orbits
    potMW2 = agama.Potential(pot_bulge2, pot_halo)   # for the forward and backford orbit integration
    potTotal= agama.Potential(potMW2, potLMCm, potacc)


    # Integrate the halo stars back in the dynamical potential including LMC
    fc_back = numpy.vstack(agama.orbit(potential=potTotal, ic=dt_ic, time=Trewind-Tcurr,
    timestart=Tcurr, trajsize=1)[:,1])
    
    # Integrate them to current position again, assuming no LMC
    fc_again = numpy.vstack(agama.orbit(potential=potMW2, ic=fc_back, time=Tcurr-Trewind,
    timestart=Trewind, trajsize=1)[:,1])
    
    chix = np.mean(fc_again[:,3])
    chiy = np.mean(fc_again[:,4])
    chiz = np.mean(fc_again[:,5])
    ll_tot = (chix**2 + chiy**2 + chiz**2)
        

# show the mean velocity vx, vy, vz as a function of radius
   
    rbin = np.array([0,5,10, 15, 20, 30, 50])
    nr2=np.size(rbin)
    rbin_plot2 = np.zeros(nr2-1)
    for i in range(nr2-1):rbin_plot2[i] = (rbin[i]+rbin[i+1])/2
    

    dt_plot = fc_again
    vx_gc = dt_plot[:,3]
    vy_gc = dt_plot[:,4]
    vz_gc = dt_plot[:,5]
    R3d_gc = np.sqrt(dt_plot[:,0]**2 + dt_plot[:,1]**2 + dt_plot[:,2]**2 )

    vx_m2 = np.zeros(nr2-1)
    vy_m2 = np.zeros(nr2-1)
    vz_m2 = np.zeros(nr2-1)

    for j in range(nr2-1):
        sk = np.where((R3d_gc > rbin[j]) & (R3d_gc < rbin[j+1] ) )
        vx_m2[j] = np.mean(vx_gc[sk])
        vy_m2[j] = np.mean(vy_gc[sk])
        vz_m2[j] = np.mean(vz_gc[sk])


    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 9}
    mpl.rc('font', **font)

    fig = plt.figure(figsize=(12,8))
    plt.subplot(2,3,1 )

    plt.plot(rbin_plot2, vx_m2, 'r--', label='vx')
    plt.plot(rbin_plot2, vy_m2, 'k--', label='vy')
    plt.plot(rbin_plot2, vz_m2, 'b--', label='vz')
    plt.legend()

    dt_plot = dt_ic
    vx_gc = dt_plot[:,3]
    vy_gc = dt_plot[:,4]
    vz_gc = dt_plot[:,5]
    R3d_gc = np.sqrt(dt_plot[:,0]**2 + dt_plot[:,1]**2 + dt_plot[:,2]**2 )
    for j in range(nr2-1):
        sk = np.where((R3d_gc > rbin[j]) & (R3d_gc < rbin[j+1] ) )
        vx_m2[j] = np.mean(vx_gc[sk])
        vy_m2[j] = np.mean(vy_gc[sk])
        vz_m2[j] = np.mean(vz_gc[sk])

    plt.plot(rbin_plot2, vx_m2, 'r-')
    plt.plot(rbin_plot2, vy_m2, 'k-')
    plt.plot(rbin_plot2, vz_m2, 'b-')

    plt.subplot(2,3,2 )
    plt.plot(trajLMC[:,1], trajLMC[:,2],'k.')
    plt.plot(trajMWx, trajMWy, 'r-')

    plt.ylim([-100,300])

    plt.subplot(2,3,2 )
    plt.plot(trajLMC[:,1], trajLMC[:,3],'k.')
    plt.plot(trajMWx, trajMWz, 'r-')

    plt.subplot(2,3,3 )
    plt.plot(trajLMC[:,2], trajLMC[:,3],'k.')
    plt.plot(trajMWy, trajMWz, 'r-')


    sfile = base_path + model + '/figure/'+ s_file_affix + '.pdf'
    plt.savefig(sfile)
    plt.close()

        
    del trajLMC, dt_ic, fc_back, fc_again
    return ll_tot
    #-----------------------------------------------------
    

