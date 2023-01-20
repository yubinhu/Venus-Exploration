# run this script from root directory
from VenusOpt.utils import get_scaler, loadXy, RBF_BEST_PARAMS, gpr_to_venus
from VenusOpt.model import generate_gpr_model
import matplotlib.pyplot as plt
import argparse
import pickle
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import time
import numpy as np

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
datafile = "Data/data%s.pkl"%exp_num if old_data else "New Data/accumulated_weekend_data_2022_12_22.h5"

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

logger = JSONLogger(path="Data/Simulations/logs_%s_%s.json" % (acq, time.strftime("%d-%m-%Y_%H-%M-%S")))
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

eps = 1e-8
cost_function = venus.cost_fn_pure

acq_func = UtilityFunction(kind=acq, kappa=2.5, xi=0.01, kappa_decay=1, kappa_decay_delay=0, cost_func=cost_function)
optimizer.maximize(init_points = 5, n_iter = n_iter, acquisition_function=acq_func, alpha=0.15)
best = optimizer.max["target"]
print("best", best)
print("time cost", venus.get_total_time())