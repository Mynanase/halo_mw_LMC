#!/usr/bin/env python
# -----------------------------------------------------------------------------
# MULTIGAUSS
#   
#   Program calculates a multivariate gaussian distribution for a given set of
#   means and covariances.  There are options to normalise the distribution or
#   adjust the height of the distibution.
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

from numpy import array, dot, exp, matrix, pi, sqrt, where
from numpy.linalg import det, inv

def multigauss( x, mu, cov, height=1., norm=True ):
    
    ndim = sqrt( cov.size )
    xdim = x.ndim
    
    if xdim == 1: x = array( [ x ] )
    
    # normalising factors
    fac = 1.
    if norm:
        fac /= sqrt( ( 2. * pi )**ndim * abs( det( cov ) ) )
    
    # invert covariance matrix
    icov = inv( cov )
    
    # calculate exponent, must be positive definite
    expo = array( [ dot( xmu, dot( icov, xmu ) ) for xmu in ( x - mu ) ] )
    expo[ where( expo < 0. ) ] = float( "nan" )
    
    # calculate multivariate gaussian
    result = exp( -0.5 * expo ) * fac * height
    if xdim == 1: result = result[0]
    
    return result
