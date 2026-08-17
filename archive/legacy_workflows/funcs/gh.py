#!/usr/bin/env python
# -----------------------------------------------------------------------------
# GH
#   
#   Gauss-Hermite series describing the expansion of a distribution around a
#   Gaussian.
#   
#   p(x) = exp(-z^2/2)/(sqrt(2*pi)*sig)*[1+sum(h_m*H_m,m=3...M)],
#   
#   where z = (x-mu)/sig. The Gaussian has mean 'mu' and standard deviation
#   'sig', and the higher order parameters h_m ( m>2, h_0=1, h_1=h_2=0 )
#   describe the deviations from this Gaussian.
#   
#   For nonzero even higher moment parameters h_2m the normalization, i.e.
#   p(x) integrated over all x (= function 'gh_integrated'), is not equal to
#   unity. Hence, for p(x) to be a probability it should be divided by this
#   normalization. 
#   
#   par = [mu,sig,h_3,h_4,...]
#   
#   HISTORY
#     v1.1 : Glenn van de Ven [glenn@strw.leidenuniv.nl] - Leiden, 2004/04
#     v1.0 : Laura L Watkins [lauralwatkins@gmail.com] - MPIA, 2013/03/01
#            - converted to python
# -----------------------------------------------------------------------------

from numpy import ones, size, where
from gaussian import gaussian
from hermite import hermite


def gh( x, par ):
    
    res = ones( x.size )
    
    if par.size > 2:
        
        z = ( x - par[0] ) / par[1]
        
        for n in range( 2, par.size ):
            res = res + par[n] * hermite( n, z )
    
    res = gaussian( x, par[0], par[1] ) * res
    
    # avoid unphysical slightly negative wings
    lim = 1e-10
    res[ where( res < lim ) ] = lim
    
    return res
