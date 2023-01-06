# NOTE: this file is outdated. See BayesOpt.py for the latest version. 

# run this script from root directory
from VenusOpt.simulator import Venus
from VenusOpt.utils import get_scaler, loadXy, RBF_BEST_PARAMS, gpr_to_venus, MATERN_BEST_PARAMS
from VenusOpt.model import generate_gpr_model
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_validate
from sklearn.utils import shuffle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import pickle

from bayes_opt import BayesianOptimization, UtilityFunction

# handle command line inputs
parser = argparse.ArgumentParser()
parser.add_argument('--run_num', type=str, default='1')
parser.add_argument('--n', type=int, default=10) 
parser.add_argument('--old_data', type=bool, default=False) 
args = parser.parse_args()
params = vars(args)

exp_num = params['run_num']
n = params['n']
old_data = params['old_data']

xcolumns=["mid_i_mean", "ext_i_mean","inj_i_mean"]
datafile = "Data/data%s.pkl"%exp_num if old_data else "New Data/accumulated_weekend_data.h5"

gpr = generate_gpr_model(exp_num, datafile=datafile, xcolumns=xcolumns, old=old_data)
x_scaler = get_scaler(xcolumns)

venus = gpr_to_venus(gpr, x_scaler)

pbounds = {"mid_i_mean": [97, 110], "ext_i_mean": [97, 110], "inj_i_mean": [116, 128]}

def try_kappa(kappa, bbf, n=10, n_iter=10):
    best_list = []
    for i in range(n):
        random_state = int(i+100*kappa)
        optimizer = BayesianOptimization(f = bbf,
                                     pbounds = pbounds, verbose = 0,
                                     random_state = random_state)
        optimizer.maximize(init_points = 5, n_iter = n_iter, kappa=kappa, alpha=0.15)
        best = optimizer.max["target"]
        best_list.append(best)
    return best_list

kappas = np.linspace(1, 8, 20)
n_iter = 30
results = []
results_dict = {}
for kappa in tqdm(kappas):
    best_list = try_kappa(kappa, venus.bbf, n=n, n_iter=n_iter)
    results_dict[kappa] = best_list
    results.append(sum(best_list)/len(best_list))

fn = "kappa_exp%s_n%d_datanorm"%(exp_num, n)

if params['old_data']:
    fn += "_olddata"

with open('Results/%s.pickle'%fn, 'wb') as file:
    pickle.dump(results_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

plt.plot(kappas, results)
plt.title("Beam current after %d steps (Exp %s)"%(n_iter, exp_num))
plt.xlabel("kappa")
plt.ylabel("Best Beam Current (averaged over %d)"%n)
plt.savefig("Graphs/%s.png"%fn)
plt.show()