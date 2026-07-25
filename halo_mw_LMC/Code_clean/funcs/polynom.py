#!/usr/bin/env python
# -----------------------------------------------------------------------------
# POLYNOM
#   
#   Calculates the a polynomial equation with given coefficients.
#   
#   INPUTS
#     x : variable coordinate
#     p : polynomial coefficients
#   
#   HISTORY
#     v1.0 : Laura L Watkins [lauralwatkins@gmail.com] - MPIA, 2012/07/30
# -----------------------------------------------------------------------------


from numpy import array, size


def polynom( x, *p ):
    
    arr = array( [ p[i] * x**i for i in range( size( p ) ) ] ).T
    y = array( [ a.sum() for a in arr ] )
    
    return y
