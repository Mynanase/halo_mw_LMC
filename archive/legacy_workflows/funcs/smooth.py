#!/usr/bin/env python
# -----------------------------------------------------------------------------
# SMOOTH
#   
#   Program smooths a given set of data points by taking a weighted average
#   over a window of five points centred on the centre point.
#   
#   INPUTS
#     y : points to be smoothed
#   
#   HISTORY
#     v1.0 : Laura L Watkins [lauralwatkins@gmail.com] - MPIA, 2012/08/22
# -----------------------------------------------------------------------------

from numpy import zeros


def smooth( y ):
    
    a = 1
    b = 1.5
    c = 2
    d = 1.5
    e = 1
    
    z = zeros( y.size )
    
    for i in range( y.size ):
        
        if i == 0 or i == y.size - 1:
            z[i] = y[i]
        elif i == 1 or i == y.size - 2:
            z[i] = ( a * y[i-1] + c * y[i] + e * y[i+1] ) / ( a + c + e )
        else:
            z[i] = ( a * y[i-2] + b * y[i-1] + c * y[i] \
                + d * y[i+1] + e * y[i+2] ) / ( a + b + c + d + e )
    
    return z