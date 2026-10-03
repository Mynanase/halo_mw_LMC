#!/usr/bin/env python
# -----------------------------------------------------------------------------
# POLYDIFF
#   
#   Calculates the derivate of a polynomial equation with given coefficients.
#   
#   INPUTS
#     x : variable coordinate
#     p : polynomial coefficients
#   
#   HISTORY
#     v1.0 : Laura L Watkins [lauralwatkins@gmail.com] - MPIA, 2012/07/30
# -----------------------------------------------------------------------------


from numpy import array, size


def polydiff( x, *p ):
    
    p = p[1:]
    
    arr = array( [ ( i + 1 ) * p[i] * x**i for i in range( size( p ) ) ] ).T
    y = array( [ a.sum() for a in arr ] )
    
    return y

