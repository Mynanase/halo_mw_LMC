# file: temp/Test_GPry.ipynb
# vi: filetype=python

# -------- code --------
from mpi4py import MPI
if MPI.COMM_WORLD.Get_rank() == 1:
    print("MPI is working")

# -------- code --------
import numpy as np
from scipy.stats import multivariate_normal

mean = [3, 2]
cov = [[0.5, 0.4], [0.4, 1.5]]
rv = multivariate_normal(mean, cov)


def logLkl(x, y):
    import time
    time.sleep(1)  # just to make the likelihood slower

    return rv.logpdf(np.array([x, y]).T)


bounds = [[-10, 10], [-10, 10]]

# -------- code --------
from gpry.run import Runner
checkpoint = "output/simple"
runner = Runner(logLkl, bounds, checkpoint=checkpoint, load_checkpoint="overwrite")

# -------- code --------
runner.run()

# -------- code --------
point = (1, 2)
print(f"Log-lkl at (1,2): {logLkl(*point)}")
print(f"surrogate at (1,2): {runner.logL(point)[0]}")

# -------- code --------
mc_samples_dict = runner.last_mc_samples()
# print(mc_samples_dict)

print('=' * 20)
