import numpy as np
from skopt import gp_minimize
from skopt import Optimizer

def f(x):
    return (np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) +
            np.random.randn() * 0.1)

opt = Optimizer([(-2.0, 2.0)])

for i in range(20):
    suggested = opt.ask()
    y = f(suggested)
    opt.tell(suggested, y)
    print('iteration:', i, suggested, y)


#res = gp_minimize(f, [(-2.0, 2.0)])


