#!/usr/bin/env python
# -----------------------------------------------------------------------------
# GAUSSIAN
#   
#   Program calculates a gaussian distribution for a given mean and standard
#   deviation.  There are options to normalise the distribution or adjust the
#   height of the distibution.
#   
#   INPUTS
#     x      : variable coordinate
#     mu     : mean of distribution
#     sigma  : standard deviation of distribution
#     height : height of distribution [default: 1]
#     norm   : option to normalise distribution [default: True]
#   
#   HISTORY
#     v1.0 : Laura L Watkins [lauralwatkins@gmail.com] - MPIA, 2012/05/21
# -----------------------------------------------------------------------------

from numpy import exp, pi, sqrt

def gaussian( x, mu, sigma, height=1., norm=True ):
    
    fac = 1.
    if norm: fac /= sqrt( 2. * pi ) * sigma
    
    result = height * fac * exp( -0.5 * ( x - mu )**2 / sigma**2 )
    
    return result
