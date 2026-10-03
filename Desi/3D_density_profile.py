import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from matplotlib.patches import Ellipse

def rho(x,y,z,r_0,p0,q0,phi0,theta0,kp1,kp2,kp3,kq1,kq2,kq3,kphi1,kphi2,kphi3,ktheta1,ktheta2,ktheta3):
    r_1,r_2,alpha_1,alpha_2,alpha_3 = 15.8041, 77.1921,  1.2795,  3.4636,  5.189
    rgc = np.sqrt(x**2 + y**2 + z**2)
    p = p0 + rgc*kp1+ rgc**2*kp2 + rgc**3*kp3
    q = q0 + rgc*kq1+ rgc**2*kq2 + rgc**3*kq3
    phi = phi0 + rgc*kphi1+ (rgc)**2*kphi2 + rgc**3*kphi3
    theta = theta0 + rgc*ktheta1+ rgc**2*ktheta2 + rgc**3*ktheta3

    
    newx = x * np.cos(phi) * np.cos(theta) - y * np.sin(phi) + z * np.cos(phi) * np.sin(theta)
    newy = x * np.sin(phi) * np.cos(theta) + y * np.cos(phi) + z * np.sin(phi) * np.sin(theta)
    newz = -x * np.sin(theta) + z * np.cos(theta)

    possi = np.vstack((newx.reshape([1,len(x)]),
                         newy.reshape([1,len(x)]),
                         newz.reshape([1,len(x)])))
    
    r = np.sqrt(possi[0,:]**2 + possi[1,:]**2/ p**2 + possi[2,:]**2/ q**2)
    
    weight = np.zeros(len(r),dtype=np.float64)

    first = (r< r_1)
    second =  (r>=  r_1) & (r<  r_2)
    third = (r>=  r_2) 
    
    weight[first] =  (r[first]/ r_1) ** (- alpha_1) 
    weight[second] = ( r_2/ r_1) ** (- alpha_2) * (r[second]/ r_2) ** (- alpha_2) 
    weight[third] =  ( r_2/ r_1) ** (- alpha_2) * (r[third]/ r_2) ** (- alpha_3)
    
    return weight

aa2 = np.array([-2.39106294e+02,  9.13810522e-01,  6.32550491e-01,  9.45689142e-01,
  1.52348376e-01, -5.28876970e-03,  6.39679000e-05, -2.55300000e-07,
  5.41388730e-03, -9.93104000e-05,  4.86200000e-07, -7.97878894e-02,
  1.25673290e-03, -6.18220000e-06, -4.52960955e-02,  3.52501300e-04,
 -3.74000000e-08])

x = np.random.uniform(-80,80,40000000)
y = np.random.uniform(-80,80,40000000)
z = np.random.uniform(-80,80,40000000)

weight1 = rho(x,y,np.zeros(len(z)),*aa2)
weight2 = rho(x,np.zeros(len(z)),z,*aa2)
weight3 = rho(np.zeros(len(z)),y,z,*aa2)

def add_hist2d_contour(ax, xdata, ydata, wdata, title):
    """hist2d + 等密度线 + 主轴"""
    # 1. 二维加权直方图
    h = ax.hist2d(xdata, ydata, weights=wdata,
                  bins=[200, 200], norm=colors.LogNorm(vmin=1, vmax=10000),
                  cmap='RdBu_r', rasterized=True)
    H, xedges, yedges = h[0], h[1], h[2]
    X, Y = np.meshgrid(0.5*(xedges[:-1]+xedges[1:]),
                        0.5*(yedges[:-1]+yedges[1:]))

    # 2. 画等高线（这里用 1,10,100,1000,10000 五条）
    levels = np.logspace(0, 3.5, 4)
    cs = ax.contour(X, Y, H.T, levels=levels, colors='red', linewidths=0.8)

    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    return h

# -------------------------------------------------
# 画图
# -------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4*3.32/3.08))   # 宽从 15 → 20

fig.subplots_adjust(wspace=0.25) 

qm1 = add_hist2d_contour(axes[0], x, z, weight2, 'x vs z')

qm2 = add_hist2d_contour(axes[1], y, z, weight3, 'y vs z')

qm3 = add_hist2d_contour(axes[2], x, y, weight1, 'x vs y')

for ax in axes:
    ax.set_yticks([-50, 0, 50])
    ax.set_yticklabels(['-50', '0', '50'])
    
axes[0].set_xlabel(r'$x \mathrm{[kpc]}$',fontsize=14)
axes[0].set_ylabel(r'$z \mathrm{[kpc]}$',fontsize=14)
axes[1].set_xlabel(r'$y \mathrm{[kpc]}$',fontsize=14)
axes[1].set_ylabel(r'$z \mathrm{[kpc]}$',fontsize=14)
axes[2].set_xlabel(r'$x \mathrm{[kpc]}$',fontsize=14)
axes[2].set_ylabel(r'$y \mathrm{[kpc]}$',fontsize=14)
axes[0].tick_params(labelsize=14)
axes[1].tick_params(labelsize=14)
axes[2].tick_params(labelsize=14)
axes[0].scatter(-8, 0, marker='*',color='gold', s=50) 
axes[1].scatter(0, 0, marker='*',color='gold', s=50)  
axes[2].scatter(-8, 0, marker='*',color='gold', s=50) 