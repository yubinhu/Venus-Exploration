import pickle
import numpy as np
import argparse

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from sklearn.model_selection import GridSearchCV

from VenusOpt.utils import loadXy

RANDSTATE = 42

# handle command line inputs
parser = argparse.ArgumentParser()
parser.add_argument('--run_num', type=str) 
args = parser.parse_args()
params = vars(args)

run_num = params['run_num']

X, y, X_var = loadXy("New Data/accumulated_weekend_data.h5", run_idx=run_num)

# hyper parameters
lmin = 0.1
lmax = 10
numin = 10
numax = 60

param_grid = [
  {'kernel__length_scale':list(np.linspace(lmin, lmax, 10)), 
   'kernel__nu':list(np.linspace(numin, numax, 10))},
]

kernel = Matern(length_scale_bounds="fixed") 
kernel_RBF = RBF(length_scale_bounds="fixed")

gpr = GaussianProcessRegressor(
    kernel=kernel, alpha=X_var.mean(), n_restarts_optimizer=9
) 

clf = GridSearchCV(estimator=gpr, param_grid=param_grid)
clf.fit(X, y)
with open("Results/gs_clf_run%s_nu%.2fto%.2f_l%.2fto%.2f.dump"%(run_num, numin, numax, lmin, lmax) , "wb") as f:
    pickle.dump(clf, f)

