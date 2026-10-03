#!/usr/bin/env python
# -----------------------------------------------------------------------------
# COV_ELLIPSE
#   
#   Program calculates the x and y coordinates of an ellipse with parameters
#   specified by a 2d covariance matrix.
#   
#   INPUTS
#     x : 2xN array of points (N = # samples)
#   
#   OPTIONS
#     sigma : level of contours (default: 1.)
#   
#   HISTORY
#     v1.0 : Laura L Watkins [lauralwatkins@gmail.com] - MPIA, 2012/06/07
# -----------------------------------------------------------------------------

from numpy import arctan2, sqrt
from numpy.linalg import eig
#from ellipse import ellipse
#from covar import covar

from numpy import cos, linspace, pi, sin, sqrt

def covar( x ):
    
    cov = array( [ [ ( i * j ).mean() - i.mean() * j.mean()
        for j in x ] for i in x ] )
    
    return cov


def ellipse( a, b, xc=0., yc=0., rot=0., n=101 ):
    
    # generate eccentric anomaly values
    ea = linspace( 0., 2. * pi, n )
    
    # x and y coordinates
    x = a * cos( ea ) * cos( rot ) + b * sin( ea ) * sin( rot ) + xc
    y = -a * cos( ea ) * sin( rot ) + b * sin( ea ) * cos( rot ) + yc
    
    return x, y


def cov_ellipse( x, sigma=1. ):
    
    cov = covar( x )
    
    # calculate eigenvalues and eigenvectors
    e, v = eig( cov )
    
    # rotation angle of the ellipse
    rot = -arctan2( v[1,0] , v[0,0] )
    
    # semi-major and semi-minor axes
    a = sqrt( e[0] ) * sigma
    b = sqrt( e[1] ) * sigma
    
    x, y = ellipse( a, b, xc=x[0].mean(), yc=x[1].mean(), rot=rot )
    
    return x, y
