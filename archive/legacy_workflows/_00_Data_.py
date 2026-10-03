import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import ticker
from matplotlib.colors import ListedColormap
from scipy import stats






def plt_scatter(fig, tk, xy, fig_cbar = False, cb = False, sb = False, tx = False, lg = False, save_nm = False):

    try:
        fig_scatter = fig[0]

        ax = plt.axes(fig[1])
        ax.tick_params(axis = tk[0], which = tk[1], direction = tk[2], length = tk[3], 
            width = tk[4], colors = tk[5], bottom = tk[6], top = tk[7], left = tk[8], 
            right = tk[9], labelbottom = tk[10], labeltop = tk[11], labelleft = tk[12], 
            labelright = tk[13])

        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(fig[2][0]))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(fig[2][1]))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(fig[2][2]))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(fig[2][3]))

        ax.xaxis.set_label_position(fig[3][0])
        ax.yaxis.set_label_position(fig[3][1])

        plt.xlim(xy[2])
        plt.ylim(xy[3])
        plt.xticks(fontsize = xy[12])
        plt.yticks(fontsize = xy[12])
        plt.xlabel(xy[4], fontsize = xy[12])
        plt.ylabel(xy[5], fontsize = xy[12])

    except:
        pass

    try:
        cs = ax.scatter(xy[0], xy[1], c = cb[0], cmap = cb[1], vmin = cb[2], vmax = cb[3],
               marker= xy[6], s = xy[8], alpha = xy[9], zorder = xy[11], rasterized = True)

        cbaxes = fig_scatter.add_axes(fig_cbar[0])


        cbar = plt.colorbar(cs, orientation = fig_cbar[1], extend = fig_cbar[2], \
            ticklocation = fig_cbar[3], cax = cbaxes, ticks = cb[4])
        cbar.set_label(label = sb[0], labelpad = sb[1], x = sb[2], y = sb[2], size = sb[3], rotation = sb[4])
        cbar.solids.set_edgecolor('face')
        cbar.ax.tick_params(labelsize = cb[5])
        
    except:
        ax.scatter(xy[0], xy[1], marker = xy[6], c = xy[7], s = xy[8], alpha = xy[9], 
            label = xy[10], zorder = xy[11], rasterized = True)

    if lg != False:
        plt.legend(loc = lg[0], shadow = True, fontsize = lg[1], frameon = 0)

    if tx != False:
        t = ax.text(tx[0], tx[1], tx[2], fontsize = tx[3], color = tx[4],
                    horizontalalignment = 'right',
                    verticalalignment = 'bottom',
                    transform = ax.transAxes)
        try:
            t.set_bbox(dict(facecolor = tx[5], alpha = tx[6], edgecolor = tx[7]))
        except:
            pass

    if save_nm != False:
        plt.savefig(r'/Users/yang/Desktop/%s'%save_nm, dpi = 300)

# plt_scatter(
#     # fig_size (plot_size, fig_size, tick_size, label_position)
#     fig = [plt.figure(figsize = (6, 4)), [0.15, 0.15, 0.70, 0.78], [40, 20, 40, 20], ['bottom', 'left']],
#     # ticks (0axis, 1which, 2direction, 3length, 4width, 5colors, 6bottom, 7top, 8left, 
#     #        9right, 10labelbottom, 11labeltop, 12labelleft, 13labelright)
#     tk = ['both', 'both', 'in', 5, 1, 'k', 1, 1, 1, 1, 1, 0, 1, 0],
#     # fig_xy (0x, 1y, 2x_range, 3y_range, 4x_label, 5y_label, 6marker, 7color, 
#     # 8size, 9alpha, 10label, 11zorder, 12fontsize)
#     xy = [data['x_gc'], data['z_gc'], [-120, 40], [-60, 60], \
#               r'$X_{\rm{gc}}$ (kpc)', r'$Z_{\rm{gc}}$ (kpc)', '.', 'k', 2, 0.5, str(), 1, 12],
#     # fig_cbar (bar_size, direction, extend, ticks_location)
#     fig_cbar = [[0.90, 0.23, 0.02, 0.65], 'vertical', 'both', 'right'],
#     # cbar (z, colormap, vmin, vmax)
#     cb = [data['feh'], 'jet', -2.0, -0.5],
#     # set_cbar (label, labelpad, x/y, fontsize, rot)
#     sb = [r'[Fe/H] (dex)', -20, -0.03, 12, 0],
#     # text (text_x, text_y, label, fontsize, color)
#     # tx = [0.18, 0.9, str(len(data)), 15, 'k'],
#     # legend (lcoation, fontsize)
#     # lg = [2, 15]
#     # save_fig
#     # save_nm = 'xy'
# )





def plt_lines(fig, tk, xy, fig_cbar = False, cb = False, sb = False, tx = False, lg = False, save_nm = False):

    try:
        fig_scatter = fig[0]

        ax = plt.axes(fig[1])
        ax.tick_params(axis = tk[0], which = tk[1], direction = tk[2], length = tk[3], 
            width = tk[4], colors = tk[5], bottom = tk[6], top = tk[7], left = tk[8], 
            right = tk[9], labelbottom = tk[10], labeltop = tk[11], labelleft = tk[12], 
            labelright = tk[13])

        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(fig[2][0]))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(fig[2][1]))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(fig[2][2]))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(fig[2][3]))

        try:
            ax.xaxis.set_label_position(fig[3][0])
            ax.yaxis.set_label_position(fig[3][1])
        except:
            pass

        plt.xlim(xy[2])
        plt.ylim(xy[3])
        plt.xticks(fontsize = xy[12])
        plt.yticks(fontsize = xy[12])
        plt.xlabel(xy[4], fontsize = xy[12])
        try:
            plt.ylabel(xy[5], fontsize = xy[12])
        except:
            pass

    except:
        pass


    plt.plot(xy[0], xy[1], linestyle = xy[6], c = xy[7], lw = xy[8], alpha = xy[9], rasterized = True, label = xy[10])

    if lg != False:
        plt.legend(loc = lg[0], shadow = True, fontsize = lg[1], frameon = 0)

    if tx != False:
        ax.text(tx[0], tx[1], tx[2], fontsize = tx[3], color = tx[4],
                horizontalalignment = 'right',
                verticalalignment = 'bottom',
                transform = ax.transAxes)

    if save_nm != False:
        plt.savefig(r'/Users/yang/Desktop/%s'%save_nm, dpi = 300)









def nu_contour(R, z, nu, nu_arr, nu_bin, th_arr, th_bin):


    cont_r = np.zeros((len(nu_arr), len(th_arr)))

    for i in range(len(nu_arr)):
        
        ix = np.where(abs(nu - (nu_arr[i])) < nu_bin)
        
        R_i = R[ix]
        z_i = z[ix]
        r_i = (R_i**2 + z_i**2)**0.5
        th_i = np.rad2deg(np.arctan(z_i/R_i))
        
        for j in range(len(th_arr)):
            jx = np.where(abs(th_i - th_arr[j]) < th_bin)
            cont_r[i][j] = np.median(r_i[jx])

    return cont_r






def bin_2d(x, y, z, s, b, r):

    v, x_edge, y_edge, bin_n = \
    stats.binned_statistic_2d(x, y, z, 
        statistic = s, bins = b, range = r)

    v_flat = v.flatten()
    v_flat[v_flat == 0] = 'nan'
    v = v_flat.reshape(b)
    v = v.T

    ############## d_Volumne ###############

    x_cen  = x_edge[:-1] + 0.5 * np.diff(x_edge)[0]
    y_cen  = y_edge[:-1] + 0.5 * np.diff(y_edge)[0]
    x_tile = np.tile(x_cen, (len(y_cen), 1))
    y_tile = np.tile(y_cen, (len(x_cen), 1)).T

    dv = (2*np.pi*(x_tile)*np.diff(x_edge)[0]**2)

    return x_edge, y_edge, x_tile, y_tile, v, dv, bin_n








def plt_dens(fig, tk, xy, fig_cbar, cb, ct, sb, tx = False, save_nm = False):

    # try:
    fig_dens = fig[0]
    ax = plt.axes(fig[1])

    ax.tick_params(axis = tk[0], which = tk[1], direction = tk[2], length = tk[3], 
        width = tk[4], colors = tk[5], bottom = tk[6], top = tk[7], left = tk[8], 
        right = tk[9], labelbottom = tk[10], labeltop = tk[11], labelleft = tk[12], 
        labelright = tk[13])

    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(fig[2][0]))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(fig[2][1]))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(fig[2][2]))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(fig[2][3]))

    ax.xaxis.set_label_position(fig[3][0])
    ax.yaxis.set_label_position(fig[3][1])

    plt.xlim(xy[2])
    plt.ylim(xy[3])
    plt.xticks(fontsize = xy[6])
    plt.yticks(fontsize = xy[6])
    plt.xlabel(xy[4], fontsize = xy[6])
    plt.ylabel(xy[5], fontsize = xy[6])

    # except:
    #     pass






    x_edge, y_edge, x_tile, y_tile, v, dv, bin_n = \
        bin_2d(xy[0], xy[1], cb[0], cb[1], cb[2], [xy[2], xy[3]])

    if cb[7] == True:
        v = v/dv

    try:
        R = np.round(x_tile.flatten(), 2)
        z = np.round(y_tile.flatten(), 2)

        cont_r = nu_contour(R, z, np.log(v.flatten()), ct[0], ct[1], ct[2], ct[3])

        ct[2] = np.deg2rad(ct[2])
        for i in range(len(ct[0])):
           plt.plot(cont_r[i]  * np.cos(ct[2]), cont_r[i] * np.sin(ct[2]), 'k-', lw = 2)
           plt.text((cont_r[i] * np.cos(ct[2]))[0]-2, (cont_r[i][0] * np.sin(ct[2]))[0]-2, 
               str(ct[0][i]), fontsize = 8)

    except:
        pass







    if cb[6] == True:
    	cs = plt.pcolormesh(x_edge, y_edge, np.log10(v), cmap = cb[3], vmin = cb[4], vmax = cb[5], edgecolors = None)
    else:
    	cs = plt.pcolormesh(x_edge, y_edge, v, cmap = cb[3], vmin = cb[4], vmax = cb[5], edgecolors = None)

    cbaxes = fig[0].add_axes(fig_cbar[0])
    cbar = plt.colorbar(cs, orientation = fig_cbar[1], extend = fig_cbar[2], ticklocation = fig_cbar[3], cax = cbaxes)
    cbar.set_label(label = sb[0], labelpad = sb[1], x = sb[2], y = sb[2], size = sb[3], rotation = sb[4])
    cbar.solids.set_edgecolor('face')

    if tx != False:
        ax.text(tx[0], tx[1], tx[2], fontsize = tx[3], color = tx[4],
                horizontalalignment = 'right',
                verticalalignment = 'bottom',
                transform = ax.transAxes)

    if save_nm != False:
        plt.savefig(r'/Users/yang/Desktop/%s'%save_nm, dpi = 300)

# plt_dens(
#     # fig_size (plot_size, fig_size, tick_size, label_position)
#     fig = [fig, fig_pos, [10, 5, 10, 5], ['bottom', 'right']],
#     # ticks (0axis, 1which, 2direction, 3length, 4width, 5colors, 6bottom, 7top, 8left, 
#     #        9right, 10labelbottom, 11labeltop, 12labelleft, 13labelright)
#     tk = ['both', 'both', 'in', 5, 1, 'k', 1, 1, 1, 1, 1, 0, 1, 0],
#     # fig_xy (0x, 1y, 2x_range, 3y_range, 4x_label, 5y_label, 6fontsize)
#     xy = [data['R'], data['z_gc'], [0, 50], [0, 50], \
#               r'$R_{\rm{gc}}$ (kpc)', r'$Z_{\rm{gc}}$ (kpc)', 12],
#     # fig_cbar (bar_size, direction, extend, ticks_locaiton)
#     fig_cbar = [[0.32, 0.92, 0.60, 0.01], 'horizontal', 'both', 'top'],
#     # cbar (0z, 1stat, 2bins, 3colormap, 4vmin, 5vmax, 6log, 7dv)
#     cb = [data['nu_ph'], 'mean', [50, 50], 'jet', -8, 0.0, True, False],
#     # contour (0cont_arr, 1nu_range, 2theta_arr, 3theta_bin)
#     ct = [[-7., -6., -5., -4., -3., -1.], 0.3, np.linspace(10, 75, 10), 15],
#     # set_cbar (label, labelpad, x/y, fontsize, rot)
#     sb = [r'ln($\nu_{\rm ph}$)', -22, -0.15, 12, 0],
#     # text (text_x, text_y, label, fontsize, color)
#     # tx = [0.18, 0.9, str(len(data)), 15, 'k'],
#     # save_fig
#     # save_nm = 'xy.pdf'
# )




def plt_hist(fig, tk, x, ht, tx = False, lg = False, save_nm = False):
    
    try:
        fig_hist = fig[0]

        ax = plt.axes(fig[1])
        ax.tick_params(axis = tk[0], which = tk[1], direction = tk[2], length = tk[3], 
            width = tk[4], colors = tk[5], bottom = tk[6], top = tk[7], left = tk[8], 
            right = tk[9], labelbottom = tk[10], labeltop = tk[11], labelleft = tk[12], 
            labelright = tk[13])

        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(fig[2][0]))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(fig[2][1]))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(fig[2][2]))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(fig[2][3]))

        ax.xaxis.set_label_position(fig[3][0])
        ax.yaxis.set_label_position(fig[3][1])

        if ht[5] == 'vertical':
            plt.xlim(x[2])
            plt.ylim(x[3])
        else:
            plt.xlim(x[3])
            plt.ylim(x[2])

        plt.xticks(fontsize = x[7])
        plt.yticks(fontsize = x[7])
        plt.xlabel(x[4], fontsize = x[7])
        plt.ylabel(x[5], fontsize = x[7])

    except:
        pass

    counts, bins = np.histogram(x[0], bins = x[1], range = x[2])
    bin_x = bins[:-1] + np.diff(bins)[0]/2

    if x[6] == 'None':
        counts = counts
    elif x[6] == 'area':
        counts = counts/np.sum(counts)
    elif x[6] == 'max':
        counts = counts/np.max(counts)
    else:
        counts, bins = np.histogram(x[0], bins = x[1], range = x[2], weights = x[6][0])
        bin_x = bins[:-1] + np.diff(bins)[0]/2
        if x[6][1] == 'None':
            counts = counts
        elif x[6][1] == 'max':
            counts = counts/np.max(counts)
        elif x[6][1] == 'area':
            counts = counts/np.sum(counts)

    plt.hist(bin_x, x[1], range = x[2], histtype = ht[0], linewidth = ht[1], linestyle = ht[2], 
        facecolor = ht[3], edgecolor = ht[3], alpha = ht[4], orientation = ht[5], label = lg[0], 
        weights = counts)

    try:
        plt.legend(loc = lg[1], shadow = True, fontsize = lg[2], frameon = 0)
    except:
        pass

    if tx != False:
        ax.text(tx[0], tx[1], tx[2], fontsize = tx[3], color = tx[4],
                horizontalalignment = 'right',
                verticalalignment = 'bottom',
                transform = ax.transAxes)

    if save_nm != False:
        plt.savefig(r'/Users/yang/Desktop/%s'%save_nm, dpi = 300)

    return bin_x, counts
        
        


# plt_hist(   
#     fig = [fig, [0.12, 0.68, 0.68, 0.30], [0.5, 0.25, 0.2, 0.1], ['bottom', 'right']],
#     # ticks (0axis, 1which, 2direction, 3length, 4width, 5colors, 6bottom, 7top, 8left, 
#     #        9right, 10labelbottom, 11labeltop, 12labelleft, 13labelright)
#     tk = ['both', 'both', 'in', 5, 1, 'k', 1, 1, 1, 1, 0, 0, 1, 0],
#     # 0x, 1bins, 2x_range, 3y_range, 4x_label, 5y_label, 6norm, 7fontsize
#     x = [data['feh'], 40, [-2.6, 0.4], [0, 1.1], str(), r'$N_{\rm norm}$', True, 12],
#     # histype, lw, ls, color, alpha, orientation
#     ht = ['step', 1.5, '-', 'r', 0.80, 'vertical'],
#     # text (text_x, text_y, label, fontsize, color)
#     # tx = [0.18, 0.9, str(len(data)), 15, 'k'],
#     # legend (label, lcoation, fontsize)
#     lg = [r'xyz', 2, 12]
#     # save_fig
#     # save_nm = 'xy'
#     )





