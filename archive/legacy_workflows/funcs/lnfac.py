#!/usr/bin/env python
# -----------------------------------------------------------------------------
# LNFAC
#   
#   Calculates the natural logarithm of a factorial - this is especially useful
#   if the factorial of large integers is required.
#   
#   HISTORY
#     v1.1 : Glenn van de Ven [glenn@strw.leidenuniv.nl] - Leiden, 2004/04
#     v1.0 : Laura L Watkins [lauralwatkins@gmail.com] - MPIA, 2013/03/01
#            - converted to python
# -----------------------------------------------------------------------------

from numpy import log, arange


def lnfac( n ):
    
    if n < 0:
        print("n should be non-negative")
        res = float( "nan" )
    
    elif n == 0: 
        res = 0.
    
    else:
        res = log( arange( 1, n ) + 1 ).sum()
    
    return res
