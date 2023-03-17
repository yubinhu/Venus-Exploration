# run this script from root directory
from VenusOpt.simulator import VenusSimulator
from VenusOpt.utils import get_scaler, loadXy, RBF_BEST_PARAMS, gpr_to_venus, MATERN_BEST_PARAMS
from VenusOpt.model import generate_gpr_model
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import pickle

from bayes_opt import BayesianOptimization

# TODO: package this up as a function

pbounds = {"A": [97, 110], "B": [97, 110], "C": [116, 128]}
random_state = 41
gpr = generate_gpr_model('1')
x_scaler = get_scaler()
venus = gpr_to_venus(gpr, x_scaler)

optimizer = BayesianOptimization(f = venus.bbf,
                                pbounds = pbounds, verbose = 0,
                                random_state = random_state)
optimizer.maximize(init_points = 5, n_iter = 30, kappa=2, alpha=0.15, **MATERN_BEST_PARAMS['1'])

print(optimizer.max)
best = optimizer.max["target"]
print(best)
