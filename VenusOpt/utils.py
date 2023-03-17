import pandas as pd
import numpy as np
from VenusOpt.simulator import VenusSimulator
import pickle

RANDSTATE = 42

RBF_BEST_PARAMS = {
    '1':{"kernel__length_scale": 0.728}, 
    '2':{"kernel__length_scale": 0.728}, 
    '3':{"kernel__length_scale": 0.728}
    }

# based on exp results on old data with datanorm on Nov17 2022
MATERN_BEST_PARAMS = {
    '1':{"kernel__length_scale": 10, "kernel__nu": 0.5}, 
    '2':{"kernel__length_scale": 10, "kernel__nu": 0.5}, 
    '3':{"kernel__length_scale": 10, "kernel__nu": 0.5}
    }

def get_scaler(xcolumns=["mid_i_mean", "ext_i_mean","inj_i_mean"], use_datanorm=False):
    norm = {
        'inj_i_mean': lambda x: (x - 116.38956255790515) / (130.30950340857873 - 116.38956255790515),
        'ext_i_mean': lambda x: (x - 96.14013084998497) / (110.50552720289964 - 96.14013084998497),
        'mid_i_mean': lambda x: (x - 93.90183492807242) / (110.00092041798128 - 93.90183492807242),
        'gas_balzer_2_mean': lambda x: (x - 10.510574340820312) / (14.524067976535894 - 10.510574340820312),
        'bias_v_mean': lambda x: (x - 9.06360577314328) / (154.53849244729068 - 9.06360577314328),
        'inj_i_std': lambda x: x / (130.30950340857873 - 116.38956255790515),
        'ext_i_std': lambda x: x / (110.50552720289964 - 96.14013084998497),
        'mid_i_std': lambda x: x / (110.00092041798128 - 93.90183492807242),
        'gas_balzer_2_std': lambda x: x / (14.524067976535894 - 10.510574340820312),
        'bias_v_std': lambda x: x / (154.53849244729068 - 9.06360577314328),
    }
    
    # norm given by marcos
    # norm = {
    #     'inj_i_mean': lambda x: (x - 116.38956255790515) / (130.30950340857873 - 116.38956255790515),
    #     'ext_i_mean': lambda x: (x - 96.14013084998497) / (110.50552720289964 - 96.14013084998497),
    #     'mid_i_mean': lambda x: (x - 93.90183492807242) / (110.00092041798128 - 93.90183492807242),
    #     'gas_balzer_2_mean': lambda x: (x - 10.510574340820312) / (14.524067976535894 - 10.510574340820312),
    #     'bias_v_mean': lambda x: (x - 9.06360577314328) / (154.53849244729068 - 9.06360577314328),
    #     'inj_i_std': lambda x: (x - 116.38956255790515) / (130.30950340857873 - 116.38956255790515),
    #     'ext_i_std': lambda x: (x - 96.14013084998497) / (110.50552720289964 - 96.14013084998497),
    #     'mid_i_std': lambda x: (x - 93.90183492807242) / (110.00092041798128 - 93.90183492807242),
    #     'gas_balzer_2_std': lambda x: (x - 10.510574340820312) / (14.524067976535894 - 10.510574340820312),
    #     'bias_v_std': lambda x: (x - 9.06360577314328) / (154.53849244729068 - 9.06360577314328),
    # }
    
    norm_from_data = {
        'inj_i_mean': lambda x: (x - 122.59401936) / 3.34305675,
        'ext_i_mean': lambda x: (x - 104.08696283) / 3.45653727,
        'mid_i_mean': lambda x: (x - 102.52640418) / 3.00584971,
        'inj_i_std': lambda x: x / 3.34305675,
        'ext_i_std': lambda x: x / 3.45653727,
        'mid_i_std': lambda x: x / 3.00584971,
        # 'gas_balzer_2_mean': lambda x: (x - 10.510574340820312) / (14.524067976535894 - 10.510574340820312),
        # 'bias_v_mean': lambda x: (x - 9.06360577314328) / (154.53849244729068 - 9.06360577314328),
    }
    
    def x_scaler(X):
        # X: ndarray with shape (N, d) or (N, )
        if len(X.shape)==1:
            X = X[None, :]
        _, d = X.shape
        assert d==len(xcolumns), "Input has feature dimension %d instead of %d"%(d, len(xcolumns))
        X_scaled = np.zeros_like(X)
        for i in range(d):
            if use_datanorm:
                X_scaled[:, i] = norm_from_data[xcolumns[i]](X[:, i])
            else:
                X_scaled[:, i] = norm[xcolumns[i]](X[:, i])
        
        return X_scaled
        
    return x_scaler

# full script for loadXy
def loadXy(data_dir, run_idx="1", xcolumns=["mid_i_mean", "ext_i_mean","inj_i_mean"], scaleX=True, ycolumns=["fcv1_i_mean"], use_datanorm=False, old=False):
    """
    Utility function for the specific data format. Load and scale. 
    :param data_dir: directory to the data files
    :return: X (num_sample,input_dim), y (num_sample,), X_var (num_sample,)
    """
    # constant definitions
    time_column = ["unix_epoch_milliseconds_mean"]
    input_columns = ['inj_i_mean', 'inj_i_std', 'ext_i_mean', 'ext_i_std', 'mid_i_mean', 'mid_i_std', 'bias_v_mean', 'bias_v_std', 'gas_balzer_2_mean', 'gas_balzer_2_std']
    output_columns = ['fcv1_i_mean', 'fcv1_i_std']
    x_std_columns = [s.replace("mean", "std") for s in xcolumns]
    cuts = {
        "1": (1645222000, 1645512000),  # coils 'inj_i', 'ext_i', 'mid_i'
        "2": (1647625000, 1647845000),  # coils 'inj_i', 'ext_i', 'mid_i'
        "3": (1651285000, 1651476000),  # coils 'inj_i', 'ext_i', 'mid_i'
        "4": (1657920000, 1658173000),  # no beam current
        "5": (1662150000, 1662490000),  # bias_v, gas_balzer_2
        "6": (1663360000, 1663570000),  # bias_v, gas_balzer_2
        "7.0": (1664580000, 1664640000),  # bias_v, gas_balzer_2
        "7.5": (1664737000, 1664810000),  # bias_v, gas_balzer_2
        "8": (1665180000, 1665405000),  # bias_v, gas_balzer_2
    }
    

    # reading data
    if old==True:
        X_unscaled, y = loadXy_old(data_dir)
        # overwrite xcolumns
        xcolumns = ["mid_i_mean", "ext_i_mean", "inj_i_mean"]
        X_std_unscaled = np.ones(X_unscaled.shape) * 0.01 # from the new data
    else:
        accumulated_data = pd.read_hdf(data_dir, "data")
        # extracting useful information
        data = accumulated_data[time_column+input_columns+output_columns]
        bounds = cuts[run_idx]
        run_data = data[(data["unix_epoch_milliseconds_mean"]/1000 > bounds[0]) & (data["unix_epoch_milliseconds_mean"]/1000 <= bounds[1])]
        
        X_unscaled = np.array(run_data[xcolumns])
        y = np.array(run_data[ycolumns]).squeeze() * 1e6
        X_std_unscaled = np.array(run_data[x_std_columns])

    if scaleX:
        x_scaler = get_scaler(xcolumns, use_datanorm=use_datanorm)
        X = x_scaler(X_unscaled)
        x_std_scaler = get_scaler(x_std_columns, use_datanorm=use_datanorm)
        X_std = x_std_scaler(X_std_unscaled)
    else:
        X = X_unscaled
        X_std = X_std_unscaled
        
    X_var = (X_std ** 2).sum(axis=1) # X_var = X1_std**2 + X2_std**2 + ...
    
    # dimension check
    assert(X.shape[0]==y.shape[0])
    assert(len(y.shape) == 1)
    assert(X.shape[0]==X_var.shape[0])

    return X, y, X_var

def loadXy_old(data_dir):
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
    X = np.array([xpts, ypts, zpts]).T
    y = np.array(magpts)
    return X, y

# converters
def gpr_to_venus(gpr, x_scaler, jitter=0.15, gpr_input_dim=3):
    # Turns a normalized gpr into venus object. 
    def unnormalized_gpr(arr):
        if gpr_input_dim>3:
            arr_zero_padded = np.pad(arr, (0, gpr_input_dim-3))
            return (gpr.predict((x_scaler(arr_zero_padded))))[0]
            
        return (gpr.predict((x_scaler(arr))))[0]
    venus = VenusSimulator(jitter=jitter, func=unnormalized_gpr)
    return venus
