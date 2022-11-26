# run this script from root directory
from VenusOpt.utils import loadXy, RBF_BEST_PARAMS, MATERN_BEST_PARAMS
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_validate
from sklearn.utils import shuffle

MATERN_KERNEL = "matern"
RBF_KERNEL = "rbf"

def generate_gpr_model(exp_num, datafile="New Data/accumulated_weekend_data.h5", 
                       kernel=MATERN_KERNEL, xcolumns=["mid_i_mean", "ext_i_mean","inj_i_mean"],
                       old=False, use_datanorm=True, verbose=0):
    X, y, X_var = loadXy(datafile, old=old, run_idx=exp_num, xcolumns=xcolumns, use_datanorm=use_datanorm)
    X, y, X_var = shuffle(X,y,X_var)

    k = Matern() if kernel==MATERN_KERNEL else RBF()
    gpr = GaussianProcessRegressor(
        kernel=k, alpha=X_var.mean(), optimizer=None
    )
    if kernel==MATERN_KERNEL:
        gpr.set_params(**MATERN_BEST_PARAMS[exp_num])
    elif kernel==RBF_KERNEL:
        gpr.set_params(**RBF_BEST_PARAMS[exp_num])
        
    if verbose>0:
        cv_score = cross_validate(gpr, X, y)["test_score"].mean()
        print("GPR model generated on exp%s with 5-fold cross-validation score %.2f"%(exp_num, cv_score))
    if verbose>1:
        print("xcolumns", xcolumns)
    gpr.fit(X, y)
    
    return gpr

