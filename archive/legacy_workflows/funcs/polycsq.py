#!/usr/bin/env python
# -----------------------------------------------------------------------------
# POLYCSQ
#   
#   Calculates the chi-squared for a polynomial with given coefficients fitted
#   to a given data set.
#   
#   INPUTS
#     p : polynomial coefficients
#     x : variable coordinate
#     y : data coordinate
#     e : errors on y
#   
#   HISTORY
#     v1.0 : Laura L Watkins [lauralwatkins@gmail.com] - MPIA, 2012/07/30
# -----------------------------------------------------------------------------


from polynom import polynom


def polycsq( p, x, y, e ):
    
    m = polynom( x, *p )
    
    csq = ( ( y - m )**2 / e**2 ).sum()
    
    return csq
