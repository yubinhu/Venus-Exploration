import pandas as pd
import numpy as np

RANDSTATE = 42

def get_scalar(xcolumns=["inj_i_mean", "ext_i_mean", "mid_i_mean"]):
    x_scaler_dict = {
        "inj_i_mean": 1 / (130-117),
        "ext_i_mean": 1 / (110-97),
        "mid_i_mean": 1 / (110-95)
    }
    x_scalar = np.array([x_scaler_dict[s] for s in xcolumns])
    return x_scalar

# full script for loadXy
def loadXy(data_dir, run_idx="1", xcolumns=["inj_i_mean", "ext_i_mean", "mid_i_mean"], ycolumns=["fcv1_i_mean"]):
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
    x_scalar = get_scalar(xcolumns)

    # reading data
    accumulated_data = pd.read_hdf(data_dir, "data")

    # extracting useful information
    data = accumulated_data[time_column+input_columns+output_columns]
    bounds = cuts[run_idx]
    run_data = data[(data["unix_epoch_milliseconds_mean"]/1000 > bounds[0]) & (data["unix_epoch_milliseconds_mean"]/1000 <= bounds[1])]
    run_data.describe()

    X = np.array(run_data[xcolumns]) * x_scalar
    y = np.array(run_data[ycolumns]).squeeze()
    X_var = (np.array(run_data[x_std_columns]) ** 2).sum(axis=1) # X_var = X1_std**2 + X2_std**2 + ...

    # dimension check
    assert(X.shape[0]==y.shape[0])
    assert(len(y.shape) == 1)
    assert(X.shape[0]==X_var.shape[0])

    return X, y, X_var
