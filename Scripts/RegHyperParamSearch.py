import pickle
import numpy as np

from CrossSecPlotter import *
from MyDataset import MyDataset

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn_evaluation import plot

RANDSTATE = 42

def loadXy(data_dir):
    """
    Utility function for the specific data format.
    :param data_dir: directory to the data files
    :return: X, y
    """

    with open(data_dir, "rb") as fp:   # Unpickling
        data_list = pickle.load(fp)

    xpts, ypts, zpts, _, magpts = data_list

    # check data
    assert(len(xpts)==len(ypts) and len(xpts)==len(zpts) and len(xpts)==len(magpts))

    # Transform the data into X, y form
    n = len(xpts)
    d = 3
    X = np.array([xpts, ypts, zpts]).T
    y = np.array(magpts)
    return X, y


datafolder = "Data/"
exp_num = 1
datafile = "data%d.pkl"%exp_num
datafile = datafolder + datafile

X, y = loadXy(datafile)

param_grid = [
  {'alpha': [0.001, 0.01, 0.1, 1], 'kernel__nu':[0.66, 0.86]},
]

gpr_ps = GaussianProcessRegressor(kernel = Matern(nu=0.66))
clf = GridSearchCV(estimator=gpr_ps,
             param_grid=param_grid)
clf.fit(X, y)

pickle.dump(clf, open("Models/HyperParamGridSearchModel.dump", 'wb'))

plot.grid_search(clf.cv_results_, change='alpha', kind='bar')