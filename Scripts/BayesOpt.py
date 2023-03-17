# run this script from root directory
from VenusOpt.utils import get_scaler, gpr_to_venus
from VenusOpt.model import generate_gpr_model
from VenusOpt.cost import CostModel
import argparse
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import time

from bayes_opt import BayesianOptimization, UtilityFunction

# handle command line inputs
parser = argparse.ArgumentParser()
parser.add_argument('--run_num', type=str, default='1')
parser.add_argument('--n_iter', type=int, default=50)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--old_data', type=bool, default=False)
parser.add_argument('--acq', type=str, default='ei') # {'ucb', 'ei', 'eipu'}
args = parser.parse_args()
params = vars(args)

exp_num = params['run_num']
old_data = params['old_data']
acq = params['acq']
n_iter = params['n_iter']
repeat = params['repeat']


# load data
xcolumns=["inj_i_mean", "mid_i_mean", "ext_i_mean"]
datafile = "Data/data%s.pkl"%exp_num if old_data else "New Data/accumulated_weekend_data_2022_12_22.h5"

# train model with best parameters
gpr = generate_gpr_model(exp_num, datafile=datafile, xcolumns=xcolumns, old=old_data)
x_scaler = get_scaler(xcolumns)


for _ in range(repeat):
    venus = gpr_to_venus(gpr, x_scaler)
    # bayes opt
    pbounds = {"mid_i_mean": [97, 110], "ext_i_mean": [97, 110], "inj_i_mean": [116, 128]}

    random_state = int(time.time())
    optimizer = BayesianOptimization(f = venus.bbf_named,
                                    pbounds = pbounds, verbose = 0,
                                    random_state = random_state, allow_duplicate_points=True)

    logfile = "Data/Simulations2/logs_%s_%s.json" % (acq, time.strftime("%d-%m-%Y_%H-%M-%S"))
    
    import json
    with open("Models/costmodel.json", "r") as f:
        POPT_DICT = json.load(f) # default cost model
    cost_model = CostModel(POPT_DICT)
    cost_function = cost_model.build_cost_function(["inj_i_mean", "mid_i_mean", "ext_i_mean"])

    acq_func = UtilityFunction(kind=acq, kappa=2, xi=0.01, kappa_decay=1, kappa_decay_delay=0, cost_func=cost_function)
    optimizer.set_gp_params(alpha=0.15, kernel__length_scale=10, kernel__nu=0.5)
    optimizer.maximize(init_points = 5, n_iter = n_iter, acquisition_function=acq_func)
    best = optimizer.max["target"]
    print("best", best)
    runtime = venus.get_total_time()
    print("time cost", venus.get_total_time())
    
    venus.dump_log(logfile)