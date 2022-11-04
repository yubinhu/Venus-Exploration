# run this script from root directory
from VenusOpt.simulator import Venus
from VenusOpt.utils import get_scalar, loadXy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_validate
from sklearn.utils import shuffle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

from bayes_opt import BayesianOptimization, UtilityFunction

# handle command line inputs
parser = argparse.ArgumentParser()
parser.add_argument('--run_num', type=str) 
parser.add_argument('--n', type=int, default=10) 
args = parser.parse_args()
params = vars(args)

exp_num = params['run_num']
n = params['n']

X, y, X_var = loadXy("New Data/accumulated_weekend_data.h5", run_idx=exp_num)
X, y = shuffle(X, y)

gpr = GaussianProcessRegressor(
    alpha=X_var.mean(), n_restarts_optimizer=9
).fit(X, y)

x_scalar = get_scalar()

#TODO: check why we need this minus sign here
unnormalized_gpr = lambda arr: (gpr.predict((arr * x_scalar).reshape(1,-1)) * 1000000)[0]
venus = Venus(jitter=0.15, func=unnormalized_gpr)

pbounds = {"A": [97, 110], "B": [97, 110], "C": [116, 128]}

def get_black_box_func(venus):
    # Define the black box function to optimize.
    def black_box_function(A, B, C):
        # C: SVC hyper parameter to optimize for.
        venus.set_mag_currents(A, B, C)
        v = venus.get_beam_current()
        return v
    return black_box_function

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

bbf = get_black_box_func(Venus(jitter=0.15, func=unnormalized_gpr))
kappas = np.linspace(1.5, 4, 10)
results = []
for kappa in tqdm(kappas):
    best_list = try_kappa(kappa, bbf, n=n)
    results.append(sum(best_list)/len(best_list))

plt.plot(kappas, results)
plt.title("Average Best Achieved (Exp %s)"%exp_num)
plt.xlabel("kappa")
plt.ylabel("Best Beam Current (averaged over %d)"%n)
plt.savefig("Graphs/kappa_exp%s_n%d.png"%(exp_num, n))
plt.show()
