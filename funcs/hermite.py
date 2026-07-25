#!/usr/bin/env python
# -----------------------------------------------------------------------------
# HERMITE
#   
#   Calculates a Hermite polynomial orthonormal w.r.t. exp(-x^2/2)
#   int_{-inf}^{inf} H_m(x) H_n(x) exp(-x^2/2)/sqrt(pi) dx = delta_{mn}
#   
#   HISTORY
#     v1.1 : Glenn van de Ven [glenn@strw.leidenuniv.nl] - Leiden, 2004/04
#     v1.0 : Laura L Watkins [lauralwatkins@gmail.com] - MPIA, 2013/03/01
#            - converted to python
# -----------------------------------------------------------------------------

from numpy import exp, sqrt
from lnfac import lnfac


def hermite( n, x ):
    
    if n < 0:
        print 'n should be non-negative'
    
    elif n == 0:
        res = 1.
    
    elif n == 1:
        res = x * sqrt( 2. )
    
    elif n == 2:
        res = ( 2. * x**2 - 1. ) / sqrt( 2. )
    
    elif n == 3:
        res = ( 2. * x**3 - 3. * x ) / sqrt( 3. )
    
    elif n == 4:
        res = ( 4. * x**4 - 12. * x**2 + 3. ) / ( 2.* sqrt( 6. ) )
    
    elif n == 5:
        res = ( 4. * x**5 - 20. * x**3 + 15. * x ) /( 2. * sqrt( 15. ) )
    
    elif n == 6:
        res = ( 8. * x**6 - 60. * x**4 + 90. * x**2 - 15. ) \
            / ( 12. * sqrt( 5. ) )
    
    else:
        res = 0.
        for j in range( int( n / 2 ) ):
            fac = exp( 0.5 * lnfac( n ) - lnfac( j ) - lnfac( n - 2 * j ) )
            res = res + fac * (-1)**j * x**( n - 2 * j ) / 4**j
        res = sqrt( 2.**n ) * res
    
    return res
