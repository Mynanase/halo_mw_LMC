#!/usr/bin/env python
# -----------------------------------------------------------------------------
# POST_PARAMS
# Laura L Watkins [lauralwatkins@gmail.com]
# -----------------------------------------------------------------------------

from numpy import *
from numpy.lib.recfunctions import append_fields
#from post_select import post_select

def nearest( x, base=1. ):
    
    """
    Round the inputs to the nearest base.  Beware, due to the nature of
    floating point arithmetic, this maybe not work as you expect.
    
    INPUTS
      x : input value of array
    
    OPTIONS
      base : number to which x should be rounded
    """
    
    return round( x / base ) * base


def whsf( x ):
    
    """
    Returns the position of the first significant figure in a floating point
    number.  Positive numbers indicate the first significant digit is after
    the decimal point, negative numbers are before the decimal point.
    
    INPUT
      x : input number
    """
    
    # calculate position of first significant figure
    sf = -int_( floor( log10( x ) ) )
    
    # round number to first significant figure
    if size( x ) == 1: rd = round( x, sf )
    else: rd = array( [ round( x[i], sf[i] ) for i in range( x.size ) ] )
    
    # check SF of rounded number as rounding can cause problems just below 1
    # e.g. 0.096 returns sf=2, but the rounded value is 0.10 which has sf=1
    sf = -int_( floor( log10( rd ) ) )
    
    return sf


def post_select( data, nsteps=0, every=2, clean=True ):
    
    if not nsteps: nsteps = data.shape[0]
    
    # select post-burn sample as every second step for a given set of steps
    post = data[-nsteps::every]
    post = post.reshape( post.size )
    
    # cut nan log likelihoods
    if clean:
        post = post[ where( ( post.ll == post.ll ) & ( post.ll != inf ) ) ]
    
    return post
    
def post_params( data, nsteps=0, every=2, prec=None, quiet=True ):
    
    """
    Calculate parameter estimates and uncertainties for an MCMC chain.
    
    INPUTS
      data   : MCMC chain
    
    OPTIONS
      nsteps : number of steps for post-burn (0 uses everything) (default 0)
      every  : plot every # steps (default 2)
      prec   : precision to quote for each parameter (default None - use 1e-10)
      quiet  : suppress read out, if set (default True)
    """
    
    
    # select post-burn sample as every second step for a given set of steps
    post = data #post_select( data, nsteps=nsteps, every=every )
    
    # make array for parameters
    names = post.dtype.names[:-1]
    nplot = array( [] )
    for name in names:
        if post[name].ptp() != 0: nplot = append( nplot, name )
    nn = nplot.size
    params = array( [] ).view( type=recarray )
    
    # mean and sigmas
    for np in nplot:
        params = append_fields( params, [ np, "e" + np ],
        [ post[np].mean(), post[np].std() ], asrecarray=True )
    
    # print parameters
    if not quiet:
        #print "-" * 30
        #print "MCMC post-burn parameters:"
        #print ""
        for i in range( size( nplot ) ):
            np = nplot[i]
            enp = "e" + np
            out = "{:>6}:  ".format( np )
            
            if prec: out += "{:7} +/- {:}".format( nearest( params[np][0],
                prec[i] ), nearest( params[enp][0], prec[i] ) )
            
            else: out += "{{:>7.{:}f}} +/- {{:<7.{:}f}}".\
                format( *[whsf( params[enp][0] )]*2 ).\
                format( params[np][0], params[enp][0] )
            
        #    print out
        #print "-" * 30
    
    return params
