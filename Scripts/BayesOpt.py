# run this script from root directory
from VenusOpt.utils import get_scaler, loadXy, RBF_BEST_PARAMS, gpr_to_venus
from VenusOpt.model import generate_gpr_model
import matplotlib.pyplot as plt
import argparse
import pickle
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import time

from bayes_opt import BayesianOptimization, UtilityFunction

# handle command line inputs
parser = argparse.ArgumentParser()
parser.add_argument('--run_num', type=str, default='1')
parser.add_argument('--n', type=int, default=10) 
parser.add_argument('--old_data', type=bool, default=False) 
parser.add_argument('--acq', type=str, default='ei') # {'ucb', 'ei'}
args = parser.parse_args()
params = vars(args)

exp_num = params['run_num']
# n = params['n']
old_data = params['old_data']
acq = params['acq']


# load data
xcolumns=["inj_i_mean", "mid_i_mean", "ext_i_mean"]
datafile = "Data/data%s.pkl"%exp_num if old_data else "New Data/accumulated_weekend_data.h5"

# train model with best parameters
gpr = generate_gpr_model(exp_num, datafile=datafile, xcolumns=xcolumns, old=old_data)
x_scaler = get_scaler(xcolumns)
venus = gpr_to_venus(gpr, x_scaler)

# bayes opt
pbounds = {"mid_i_mean": [97, 110], "ext_i_mean": [97, 110], "inj_i_mean": [116, 128]}
n_iter = 30

random_state = int(time.time())
optimizer = BayesianOptimization(f = venus.bbf_named,
                                pbounds = pbounds, verbose = 2,
                                random_state = random_state)

logger = JSONLogger(path="Data/Simulations/logs.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)


optimizer.maximize(init_points = 5, n_iter = n_iter, acq=acq, xi=0.01, kappa=2.5, alpha=0.15)
best = optimizer.max["target"]
print("best", best)