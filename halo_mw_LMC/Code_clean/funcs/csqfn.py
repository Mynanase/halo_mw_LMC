#!/usr/bin/env python
# -----------------------------------------------------------------------------
# CSQFN
#   
#   Program computes chi-squared for a given function.
#   
#   INPUTS
#     p  : model parameters
#     x  : variable coordinates
#     y  : data values at x
#     e  : errors on y
#     fn : model function, must be callable by as fn( x, *p )
#   
#   HISTORY
#     v1.0 : Laura L Watkins [lauralwatkins@gmail.com] - MPIA, 2012/08/21
# -----------------------------------------------------------------------------


def csqfn( p, x, y, e, fn ):
    
    m = fn( x, *p )
    
    csq = ( ( y - m )**2 / e**2 ).sum()
    
    return csq
