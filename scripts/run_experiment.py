#!/usr/bin/env python
# coding: utf-8
"""Grid-scan the legacy LMC backward/forward closure objective over massLMC and halo shape.

Question: the round-halo smoke scan (qhalo=phalo=1.0, gamma=1.0, rho0=6.5,
rs=1.65) found an interior minimum of int_one_LMC near massLMC=11.3, which
coincides with the archived prior's upper edge Real(11.0, 11.3). This run
tests whether that minimum survives changes to the halo shape parameters, or
whether its location is an artifact of the fixed spherical halo.

Each objective call prints a GRID line; each halo setting ends with a SUMMARY
line reporting the argmin mass and whether it is interior to the scanned
range; the run ends with DONE. The evaluator is scripts/lmc_back_evaluator.py,
a copy of archive/legacy_workflows/Bayes_oint_LMC_back.py with exactly two
substitutions for the vendored agama 1.0 build (see its header and AGENTS.md).
"""

import argparse
import csv
import os
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--catalogue',
                    default='/home/tqiu/halo_mw_LMC/data_for_model/lamost_dr8_SFlast_cut4_4phi/halo_clean_N.txt')
    ap.add_argument('--figdir', default='lmc_grid_figs')
    ap.add_argument('--csv', default='lmc_mass_grid.csv')
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    from lmc_back_evaluator import int_one_LMC

    # int_one_LMC writes one diagnostic figure per call under base_path+model+'/figure/'
    base = os.getcwd() + os.sep
    os.makedirs(os.path.join(base, args.figdir, 'figure'), exist_ok=True)

    rho0, rs = 6.5, 1.65
    masses = [10.8, 10.9, 11.0, 11.1, 11.15, 11.2, 11.3, 11.4, 11.5]
    halo_settings = [dict(qhalo=q, phalo=p, gamma=1.0)
                     for q in (0.8, 1.0, 1.2) for p in (0.8, 1.0, 1.2)]
    halo_settings += [dict(qhalo=1.0, phalo=1.0, gamma=g) for g in (0.5, 1.5)]
    print('CONFIG catalogue=%s rho0=%.2f rs=%.2f masses=%s settings=%d'
          % (args.catalogue, rho0, rs, masses, len(halo_settings)), flush=True)

    rows = []
    t_start = time.time()
    for hs in halo_settings:
        for mass in masses:
            t0 = time.time()
            ll = int_one_LMC(base, args.figdir, rho0, rs, hs['phalo'], hs['qhalo'],
                             0, 0, hs['gamma'], mass, args.catalogue)
            rows.append((hs['qhalo'], hs['phalo'], hs['gamma'], mass, ll))
            print('GRID qhalo=%.2f phalo=%.2f gamma=%.2f massLMC=%.2f ll_tot=%.6f dt=%.1fs'
                  % (hs['qhalo'], hs['phalo'], hs['gamma'], mass, ll, time.time() - t0), flush=True)
        sub = [r for r in rows if r[0:3] == (hs['qhalo'], hs['phalo'], hs['gamma'])]
        best = min(sub, key=lambda r: r[4])
        interior = masses[0] < best[3] < masses[-1]
        print('SUMMARY qhalo=%.2f phalo=%.2f gamma=%.2f best_mass=%.2f best_ll=%.6f interior=%s'
              % (hs['qhalo'], hs['phalo'], hs['gamma'], best[3], best[4], interior), flush=True)

    with open(args.csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['qhalo', 'phalo', 'gamma', 'massLMC', 'll_tot'])
        w.writerows(rows)

    per_setting_best = [
        min([r for r in rows if r[0:3] == (hs['qhalo'], hs['phalo'], hs['gamma'])], key=lambda r: r[4])
        for hs in halo_settings]
    n_interior = sum(1 for b in per_setting_best if masses[0] < b[3] < masses[-1])
    print('DONE calls=%d settings=%d interior_argmin=%d/%d elapsed=%.1fs'
          % (len(rows), len(halo_settings), n_interior, len(halo_settings), time.time() - t_start), flush=True)


if __name__ == '__main__':
    main()
