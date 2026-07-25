#!/usr/bin/env python
# -----------------------------------------------------------------------------
# ASYMGAUSS
#   
#   Program calculates an asymmetric gaussian distribution (that is, a gaussian
#   with different standard deviations above and below the mean) for a given
#   mean and standard deviations.  There are options to normalise the
#   distribution or adjust the height of the distibution.
#   
#   INPUTS
#     x      : variable coordinate
#     mu     : mean of distribution
#     sig_lo : standard deviation of distribution below the mean
#     sig_hi : standard deviation of distribution above the mean
#     height : height of distribution [default: 1]
#     norm   : option to normalise distribution [default: True]
#   
#   HISTORY
#     v1.0 : Laura L Watkins [lauralwatkins@gmail.com] - MPIA, 2012/05/21
# -----------------------------------------------------------------------------

from numpy import array, pi, size, sqrt, where, zeros
from gaussian import gaussian


def asymgauss( x, mu, sig_lo, sig_hi, height=1., norm=True ):
    
    if size( x ) == 1 and type( x ) == float:
        flag = True
        x = array( [ x ] )
    else: flag = False
    
    fac = 1.
    if norm: fac /= sqrt( pi / 2. ) * ( abs( sig_lo ) + abs( sig_hi ) )
    
    y_lo = gaussian( x, mu, abs( sig_lo ), norm=False )
    y_hi = gaussian( x, mu, abs( sig_hi ), norm=False )
    
    f_lo = zeros( x.shape )
    f_lo[ where( x <= mu ) ] = 1
    f_hi = zeros( x.shape )
    f_hi[ where( x > mu ) ] = 1
    
    y = f_lo * y_lo + f_hi * y_hi
    
    result = height * fac * y
    
    if flag: result = result[0]
    
    return result
