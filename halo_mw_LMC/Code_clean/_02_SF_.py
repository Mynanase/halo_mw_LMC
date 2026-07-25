import numpy as np

from scipy import stats
from scipy.interpolate import interp1d
from tqdm import tqdm
from joblib import Parallel, delayed



def norm_pdf(x, mu, sigma):

    return np.exp(-(0.5)*((x - mu)/sigma)**2)


def dirac(x, mu, sigma):
    
    y = np.zeros(len(x))
    y[abs(x - mu) <= sigma] = 1

    return y



# def nu_i_lc(i, SF_m, SF_i, PI, MK, JK, Dt, DL, DU, SF):
    
#     dd = 0.01; dm = 0.25; dc = 0.1
#     dg = np.round(np.arange(0.1, 100. + dd, dd), 2) # d_grid
#     cg = np.round(np.arange(-0.5,  4. + dc, dc), 2) # c_grid
#     mg = np.round(np.arange(0.,   16. + dm, dm), 2) # m_grid
#     Om = 20./(4*np.pi * (180/np.pi)**2) 
#     # plate area = 20 (deg^2), 
#     # spherical area = 4*np.pi * (180/np.pi)**2 (deg^2)

#     pi_i = np.where(PI == SF_m[i])[0] # same_plateid_stars

#     c  = JK[pi_i]; m  = MK[pi_i]; d = Dt[pi_i]
#     dl = DL[pi_i]; du = DU[pi_i]

#     ci = np.array([int(i) for i in np.round((c - cg[0])/dc)])
#     mi = np.array([int(i) for i in np.round((m - mg[0])/dm)])

#     sf_i = SF[SF_i[i], 1:]
#     # sf_ij = sf_i[ci + mi*len(cg)]
#     sf_ij = sf_i[mi + ci*len(mg)]

#     nu_sp_d = np.zeros(len(dg))
#     nu_ph_d = np.zeros(len(dg))
#     nm_dmdc = np.zeros(len(dg))
    
#     for j in range(len(sf_ij)):

#         p_dl = norm_pdf(dg, d[j], 1.0)
#         p_du = norm_pdf(dg, d[j], 1.0)
#         prob = p_dl.copy()
#         prob[dg > d[j]] = p_du[dg > d[j]]

#         # nu_sp_j = prob/np.sum(prob)
#         # nu_sp_j = dirac(dg, np.floor(d[j]) + 0.5, 0.49)
#         # nu_sp_j = (Om * dg**2)**(-1) * (prob/np.sum(prob)) / len(sf_ij)
#         nu_sp_j = ((Om * dg**2)**(-1) * prob/np.sum(prob)) * dm * dc
#         nu_ph_j = nu_sp_j * sf_ij[j]

#         nu_sp_d = nu_sp_d + nu_sp_j
#         nu_ph_d = nu_ph_d + nu_ph_j

#         #nm_dmdc = nm_dmdc + dm * dc

#     #nu_sp_d = nu_sp_d/nm_dmdc
#     #nu_ph_d = nu_ph_d/nm_dmdc

#     nu_sp = interp1d(dg, nu_sp_d, bounds_error = 0, fill_value = 0)(np.array(Dt[pi_i]))
#     nu_ph = interp1d(dg, nu_ph_d, bounds_error = 0, fill_value = 0)(np.array(Dt[pi_i]))

#     return pi_i, nu_sp, nu_ph, nu_sp_d, nu_ph_d, sf_ij


def nu_i_lc(i, SF_m, SF_i, PI, MK, JK, Dt, DL, DU, SF):

    dd = 0.01; dm = 0.25; dc = 0.1
    dg = np.round(np.arange(0.1, 100. + dd, dd), 2) # d_grid
    mg = np.round(np.arange(0.,   15. + dm, dm), 2) # m_grid
    cg = np.round(np.arange(-0.5,  4. + dc, dc), 2) # c_grid
    Om = 20./(4*np.pi * (180/np.pi)**2)
    # plate area = 20 (deg^2),
    # spherical area = 4*np.pi * (180/np.pi)**2 (deg^2)

    pi_i = np.where(PI == SF_m[i])[0] # same_plateid_stars

    m  = MK[pi_i]; c  = JK[pi_i]; d = Dt[pi_i]
    dl = DL[pi_i]; du = DU[pi_i]

    mi = np.array([int(i) for i in np.round((m - mg[0])/dm)])
    ci = np.array([int(i) for i in np.round((c - cg[0])/dc)])

    sf_i = SF[SF_i[i], 1:]
    sf_ij = sf_i[mi + ci*len(mg)]

    nu_sp_d = np.zeros(len(dg))
    nu_ph_d = np.zeros(len(dg))

    for j in range(len(sf_ij)):

        p_dl = norm_pdf(dg, d[j], dl[j])
        p_du = norm_pdf(dg, d[j], du[j])
        prob = p_dl.copy()
        prob[dg > d[j]] = p_du[dg > d[j]]

        nu_sp_j = (Om * dg**2)**(-1) * (prob/np.sum(prob))# * dm * dc
        nu_ph_j = nu_sp_j * sf_ij[j]

        nu_sp_d = nu_sp_d + nu_sp_j
        nu_ph_d = nu_ph_d + nu_ph_j

    nu_sp = interp1d(dg, nu_sp_d, bounds_error = 0, fill_value = 0)(np.array(Dt[pi_i]))
    nu_ph = interp1d(dg, nu_ph_d, bounds_error = 0, fill_value = 0)(np.array(Dt[pi_i]))

    return pi_i, nu_sp, nu_ph, nu_sp_d, nu_ph_d, sf_ij





def nu_i(i, SF_m, SF_i, PI, MK, JK, Dt, DL, DU, SF):
    
    dd = 0.01; dm = 0.25; dc = 0.1
    dg = np.round(np.arange(0.1,  100. + dd, dd), 2) # d_grid
    cg = np.round(np.arange(-0.6,   4.,      dc), 2) # c_grid
    mg = np.round(np.arange(-0.25, 16.,      dm), 2) # m_grid
    Om = 20./(4*np.pi * (180/np.pi)**2) 
    # plate area = 20 (deg^2), 
    # spherical area = 4*np.pi * (180/np.pi)**2 (deg^2)

    pi_i = np.where(PI == SF_m[i])[0] # same_plateid_stars

    c  = JK[pi_i]; m  = MK[pi_i]; d = Dt[pi_i]
    dl = DL[pi_i]; du = DU[pi_i]

    ci = np.array([int(i) for i in np.floor((c - cg[0])/dc)])
    mi = np.array([int(i) for i in np.floor((m - mg[0])/dm)])

    sf_i = SF[SF_i[i], 1:]
    sf_ij = sf_i[ci + mi*len(cg)]

    nu_sp_d = np.zeros(len(dg))
    nu_ph_d = np.zeros(len(dg))
    nm_dmdc = np.zeros(len(dg))
    
    for j in range(len(sf_ij)):

        p_dl = norm_pdf(dg, d[j], 1.0)
        p_du = norm_pdf(dg, d[j], 1.0)
        prob = p_dl.copy()
        prob[dg > d[j]] = p_du[dg > d[j]]

        # nu_sp_j = prob/np.sum(prob)
        # nu_sp_j = dirac(dg, np.floor(d[j]) + 0.5, 0.49)
        # nu_sp_j = (Om * dg**2)**(-1) * (prob/np.sum(prob)) / len(sf_ij)
        nu_sp_j = ((Om * dg**2)**(-1) * prob/np.sum(prob))# * dm * dc
        nu_ph_j = nu_sp_j * sf_ij[j]

        nu_sp_d = nu_sp_d + nu_sp_j
        if ~np.isinf(sf_ij[j]):
            nu_ph_d = nu_ph_d + nu_ph_j
        # nu_ph_d = nu_ph_d + nu_ph_j

        #nm_dmdc = nm_dmdc + dm * dc

    #nu_sp_d = nu_sp_d/nm_dmdc
    #nu_ph_d = nu_ph_d/nm_dmdc

    nu_sp = interp1d(dg, nu_sp_d, bounds_error = 0, fill_value = 0)(np.array(Dt[pi_i]))
    nu_ph = interp1d(dg, nu_ph_d, bounds_error = 0, fill_value = 0)(np.array(Dt[pi_i]))

    return pi_i, nu_sp, nu_ph, nu_sp_d, nu_ph_d, sf_ij






def nu(SF_m, SF_i, PI, MK, JK, Dt, DL, DU, SF, n_jobs, lc = False):

    if lc == True:
        nu_ = np.array(Parallel(n_jobs = n_jobs)(delayed(nu_i_lc)\
            (i, SF_m, SF_i, PI, MK, JK, Dt, DL, DU, SF) for i in \
            tqdm(range(len(SF_m)), position = 0, leave = True, \
                ascii = True, ncols = 80, desc = 'Runing')), dtype=object)
    else:
        nu_ = np.array(Parallel(n_jobs = n_jobs)(delayed(nu_i)\
            (i, SF_m, SF_i, PI, MK, JK, Dt, DL, DU, SF) for i in \
            tqdm(range(len(SF_m)), position = 0, leave = True, \
                ascii = True, ncols = 80, desc = 'Runing')), dtype=object)


    pi_i    = np.concatenate([nu_[i][0] for i in range(len(nu_))]).argsort()
    nu_sp   = np.concatenate([nu_[i][1] for i in range(len(nu_))])
    nu_ph   = np.concatenate([nu_[i][2] for i in range(len(nu_))])
    sf_ij   = np.concatenate([nu_[i][5] for i in range(len(nu_))])
    nu_sp_d =       np.array([nu_[i][3] for i in range(len(nu_))])
    nu_ph_d =       np.array([nu_[i][4] for i in range(len(nu_))])

    return nu_sp[pi_i], nu_ph[pi_i], sf_ij[pi_i]#, nu_sp_d, nu_ph_d













def w(R, z, nu_ph, bins, x_range, y_range):

	nu_ph_stats = stats.binned_statistic_2d(R, z, nu_ph, statistic = 'mean',  
	                                        bins = bins, range = (x_range, y_range))
	nu_sp_stats = stats.binned_statistic_2d(R, z, nu_ph, statistic = 'count', 
	                                        bins = bins, range = (x_range, y_range))

	############## d_Volumne ###############
	x_edge = nu_ph_stats[1]; y_edge = nu_ph_stats[2]
	x_cen  = x_edge[:-1] + 0.5 * np.diff(x_edge)[0]
	y_cen  = y_edge[:-1] + 0.5 * np.diff(y_edge)[0]
	x_tile = np.tile(x_cen, (len(y_cen),1))

	dV = 2*np.pi*(x_tile)*np.diff(x_edge)[0]*np.diff(y_edge)[0]
	dV = np.pad(dV, ((1,1), (1,1)), 'constant', constant_values = 0).T

	################# n_ph #################
	av_nu_ph = nu_ph_stats[0]
	av_nu_ph = np.pad(av_nu_ph, ((1,1),(1,1)), 'constant', constant_values = 0)
	n_ph = av_nu_ph * dV

	################# n_sp #################
	n_sp = nu_sp_stats[0]
	n_sp = np.pad(n_sp, ((1,1),(1,1)), 'constant', constant_values = 0)

	############### bin_index ##############
	bin_n = nu_sp_stats[-1]

	wi = (n_ph/n_sp).flatten()[bin_n]

	return wi















def nu_ph_(k, n, PI, MK, JK, Dt_tl, DL, DU, SF):

    Dt = Dt_tl[k::n]

    dd = 0.01; dm = 0.25; dc = 0.1
    dg = np.round(np.arange(0.1, 100. + dd, dd), 2) # d_grid
    mg = np.round(np.arange(0.,   15. + dm, dm), 2) # m_grid
    cg = np.round(np.arange(-0.5,  4. + dc, dc), 2) # c_grid
    Om = 20./(4*np.pi * (180/np.pi)**2) 
    # plate area = 20 (deg^2), 
    # spherical area = 4*np.pi * (180/np.pi)**2 (deg^2)

    nu_sp_i = np.zeros((len(SF), len(dg)))
    nu_ph_i = np.zeros((len(SF), len(dg)))
    nu_sp = np.zeros(len(Dt))
    nu_ph = np.zeros(len(Dt))

    for i in range(len(SF)):

        pi_i = np.where((PI == SF[:, 0][i]))[0] # same_plateid_stars

        if len(pi_i) > 0:

            m  = MK[pi_i]; c  = JK[pi_i]; d  = Dt[pi_i]
            dl = DL[pi_i]; du = DU[pi_i]

            mi = np.array([int(i) for i in np.round((m - mg[0])/dm)])
            ci = np.array([int(i) for i in np.round((c - cg[0])/dc)])

            sf_i = SF[i, 1:]
            sf_ij = sf_i[mi + ci*len(mg)]

            for j in range(len(sf_ij)):

                p_dl = norm_pdf(dg, d[j], dl[j])
                p_du = norm_pdf(dg, d[j], du[j])
                prob = p_dl.copy()
                prob[dg > d[j]] = p_du[dg > d[j]]

                nu_sp_j = (Om * dg**2)**(-1) * (prob/np.sum(prob))# * dm * dc
                nu_ph_j = nu_sp_j * sf_ij[j]

                nu_sp_i[i] = nu_sp_i[i] + nu_sp_j
                nu_ph_i[i] = nu_ph_i[i] + nu_ph_j

            nu_sp[pi_i] = interp1d(dg, nu_sp_i[i], bounds_error = 0, fill_value = 0)(np.array(Dt[pi_i]))
            nu_ph[pi_i] = interp1d(dg, nu_ph_i[i], bounds_error = 0, fill_value = 0)(np.array(Dt[pi_i]))

    return nu_sp, nu_ph# , nu_sp_i, nu_ph_i



def nu_n(PI, MK, JK, Dt_tl, DL, DU, SF, n, n_jobs):

    nu_ = np.array(Parallel(n_jobs = n_jobs)(delayed(nu_ph_)\
        (k, n, PI, MK, JK, Dt_tl, DL, DU, SF) for k in tqdm(range(n), \
            position = 0, leave = True, ascii = True, ncols = 80, desc = 'Runing')), dtype = float)

    nu_sp = (np.array([nu_[i][0] for i in range(len(nu_))]).T).flatten()
    nu_ph = (np.array([nu_[i][1] for i in range(len(nu_))]).T).flatten()

    

    return nu_sp, nu_ph














# def nu_ph(PI, MK, JK, Dt, DL, DU, SF):

#     dd = 0.01; dm = 0.25; dc = 0.1
#     dg = np.round(np.arange(0.1, 100. + dd, dd), 2) # d_grid
#     mg = np.round(np.arange(0.,   15. + dm, dm), 2) # m_grid
#     cg = np.round(np.arange(-0.5,  4. + dc, dc), 2) # c_grid
#     Om = 20./(4*np.pi * (180/np.pi)**2) 
#     # plate area = 20 (deg^2), 
#     # spherical area = 4*np.pi * (180/np.pi)**2 (deg^2)

#     nu_sp_i = np.zeros((len(SF), len(dg)))
#     nu_ph_i = np.zeros((len(SF), len(dg)))
#     nu_sp = np.zeros(len(Dt))
#     nu_ph = np.zeros(len(Dt))

#     for i in tqdm(range(len(SF)), ascii = True, ncols = 60, desc = 'Plate...'):

#         pi_i = np.where((PI == SF[:, 0][i]))[0] # same_plateid_stars

#         if len(pi_i) > 0:

#             m  = MK[pi_i]; c  = JK[pi_i]; d  = Dt[pi_i]
#             dl = DL[pi_i]; du = DU[pi_i]

#             mi = np.array([np.int(i) for i in np.round((m - mg[0])/dm)])
#             ci = np.array([np.int(i) for i in np.round((c - cg[0])/dc)])

#             sf_i = SF[i, 1:]
#             sf_ij = sf_i[mi + ci*len(mg)]

#             for j in range(len(sf_ij)):

#                 p_dl = norm_pdf(dg, d[j], dl[j])
#                 p_du = norm_pdf(dg, d[j], du[j])
#                 prob = p_dl.copy()
#                 prob[dg > d[j]] = p_du[dg > d[j]]

#                 nu_sp_j = (Om * dg**2)**(-1) * (prob/np.sum(prob)) * dm * dc
#                 nu_ph_j = nu_sp_j * sf_ij[j]

#                 nu_sp_i[i] = nu_sp_i[i] + nu_sp_j
#                 nu_ph_i[i] = nu_ph_i[i] + nu_ph_j

#             nu_sp[pi_i] = interp1d(dg, nu_sp_i[i], bounds_error = 0, fill_value = 0)(np.array(Dt[pi_i]))
#             nu_ph[pi_i] = interp1d(dg, nu_ph_i[i], bounds_error = 0, fill_value = 0)(np.array(Dt[pi_i]))

#     return nu_sp, nu_ph, nu_sp_i, nu_ph_i


# sf_path = basepath + '/Data/LiuC/SF/LM5/'

# sf_cm  = np.array(table.Table.read(sf_path + r'sf.fits'    )['sf' ])
# sf_pid = np.array(table.Table.read(sf_path + r'sf_pid.fits')['pid'])

# sf_cm = sf_cm.reshape((4147, 2806))
# SF  = np.c_[sf_pid, sf_cm]

# PI = np.array(data['pid'])
# MK = np.array(data['Kmag'])
# JK = np.array(data['Jmag'] - data['Kmag'])
# Dt = np.array(data['d'])
# DL = DU = np.array(data['d_err'])

# data['nu_sp'], data['nu_ph'], _, _ = nu_ph(PI, MK, JK, Dt, DL, DU, SF)
# del PI, MK, JK, Dt, DL, DU


