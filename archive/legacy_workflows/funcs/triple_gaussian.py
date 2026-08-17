#!/usr/bin/env python
# -----------------------------------------------------------------------------
# TRIPLE_GAUSSIAN
#   
#   Program calculates a triple gaussian distribution for a given means and
#   standard deviations and relative contributions.
#   
#   INPUTS
#     x  : variable coordinate
#     m1 : mean of first gaussian
#     s1 : standard deviation of first gaussian
#     h1 : height of first gaussian
#     m2 : mean of second gaussian
#     s2 : standard deviation of second gaussian
#     h2 : height of second gaussian
#     m3 : mean of third gaussian
#     s3 : standard deviation of third gaussian
#     h3 : height of third gaussian
#   
#   HISTORY
#     v1.0 : Laura L Watkins [lauralwatkins@gmail.com] - MPIA, 2012/08/21
# -----------------------------------------------------------------------------

from gaussian import gaussian


def triple_gaussian( x, m1, s1, h1, m2, s2, h2, m3, s3, h3 ):
    
    y = gaussian( x, m1, s1, norm=False, height=h1 ) \
        + gaussian( x, m2, s2, norm=False, height=h2 ) \
        + gaussian( x, m3, s3, norm=False, height=h3 )
    
    return y
