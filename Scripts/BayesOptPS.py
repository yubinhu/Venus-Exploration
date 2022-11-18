# run this script from root directory
from VenusOpt.simulator import Venus
from VenusOpt.utils import get_scaler, loadXy, RBF_BEST_PARAMS, gpr_to_venus, MATERN_BEST_PARAMS
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

if params['old_gpr']:
    with open("Models/gprMatern0.66.dump", "rb") as file:
        gpr = pickle.load(file) # based on the fact that the scalar is the same
    x_scaler = get_scaler()
else:
    X, y, X_var = loadXy("New Data/accumulated_weekend_data.h5", 
                        run_idx=exp_num, use_datanorm=True)
    X, y = shuffle(X, y)
    
old_data = params['old_data']
datafile = "../Data/data%s.pkl"%exp_num if old_data else "../New Data/accumulated_weekend_data.h5"
X, y, X_var = loadXy(datafile, old=old_data, run_idx=exp_num)
X, y, X_var = shuffle(X,y,X_var)

gpr = GaussianProcessRegressor(
    kernel=Matern(), alpha=X_var.mean(), optimizer=None
)
gpr.set_params(**MATERN_BEST_PARAMS[exp_num])
gpr.fit(X, y)

x_scaler = get_scaler()
venus = gpr_to_venus(gpr, x_scaler)

pbounds = {"A": [97, 110], "B": [97, 110], "C": [116, 128]}

def try_kappa(kappa, bbf, n=10):
    best_list = []
    for i in range(n):
        random_state = int(i+100*kappa)
        optimizer = BayesianOptimization(f = bbf,
                                     pbounds = pbounds, verbose = 0,
                                     random_state = random_state)
        optimizer.maximize(init_points = 5, n_iter = 30, kappa=kappa, alpha=0.15)
        best = optimizer.max["target"]
        best_list.append(best)
    return best_list

kappas = np.linspace(1, 8, 20)
results = []
results_dict = {}
for kappa in tqdm(kappas):
    best_list = try_kappa(kappa, venus.bbf, n=n)
    results_dict[kappa] = best_list
    results.append(sum(best_list)/len(best_list))

fn = "kappa_exp%s_n%d_datanorm"%(exp_num, n)

if params['old_data']:
    fn += "_olddata"

with open('Results/%s.pickle'%fn, 'wb') as file:
    pickle.dump(results_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

plt.plot(kappas, results)
plt.title("Average Best Achieved (Exp %s)"%exp_num)
plt.xlabel("kappa")
plt.ylabel("Best Beam Current (averaged over %d)"%n)
plt.savefig("Graphs%fn.png"%fn)
plt.show()
