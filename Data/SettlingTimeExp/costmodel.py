import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.optimize import curve_fit
import glob

plt.rcParams['figure.dpi']=150
plt.rcParams['font.family'] = 'Serif'

# assume the last LAST_N datapoints are stable
LAST_N = 500
DATAFOLDER = "Data/SettlingTimeExp/datafiles/"
IMAGEFOLDER = "Graphs/SettlingTime/"
dataset_dict = {"inj" : 3, "ext": 4, "mid": 5}

def load_tracking(name):
    with open(name) as datafile:
        content = datafile.readlines()
        data = np.array([np.fromstring(c, sep=' ') for c in content])
    return data[:,0],data[:,1:]

# raw data...no time averaging
def get_raw_data(dataset_num, yset = 2):
    """ get raw beam current data.
    dataset_number: 0:balzer, 1:28GHz, 2:18GHz 3:inj 4:ext 5:mid 6:sext 7:bias
    yset:
    2:Ibeam 25:Pinj
    68 set balz 1
    69 set balz 2
    70 set balz 5
    71 set balz 6
    72 set balz 7
    73 set I$_\mathrm{inj}$ [A]
    74 set I$_\mathrm{ext}$ [A]
    75 set I$_\mathrm{mid}$ [A]
    76 set I$_\mathrm{sext}$ [A]
    77 set P$_\mathrm{18}$ [W]
    78 set P$_\mathrm{28}$ [W]
    normalize: 0: not normalized, 1: normalized to t=0 value, 2: normalized to t[-1]-t[t=0] value 3: set values
    """
    filest = ['balzer_*','p28_*','p18_*','inj_*','ext_*','mid_*','sext_*',"bias_*"]
    files = []
    for file in glob.glob(DATAFOLDER+filest[dataset_num]):
        files.append(file)
    files.sort()
    
    normalize = 0
      # 0: not normalized, 1: normalized to t=0 value, 2: normalized to t[-1]-t[t=0] value
    time_series = []
    for i in range(len(files)):
        time,data = load_tracking(files[i])
        X = time-time[49]
        if normalize ==0:
            time_series.append((X, (data[:,yset]-data[0,yset])*1.0e6))
        if normalize ==1:
            time_series.append((X, (data[:,yset]-data[49,yset])/data[49,yset]*100))
        if normalize ==2:
            time_series.append((X, (data[:,yset]-data[49,yset])/(np.mean(data[-50:,yset])-data[49,yset])))
            # plt.ylim([-.1,1.1])
        if normalize ==3:
            time_series.append((X, data[:,yset]-data[0,yset]))
            
    return time_series

def get_set_values(set_data):
    # get_set_values for the coils
    set_values = []
    for x,y in set_data:
        set_values.append(y[500] * 1e-6) # get rid of normalization
    return set_values

def stabilization_time(data, start_i=49):
    settle_times = []
    for x, y in data:
        y = medfilt(y, kernel_size=9)
        center = np.mean(y[-LAST_N:])
        std = np.std(y[-LAST_N:])
        ul = center + 6*std
        ll = center - 6*std
        bounds = (ll, ul)
        settle_time = -1
        for i, yi in reversed(list(enumerate(y))[start_i:-LAST_N]):
            if yi < bounds[0] or yi > bounds[1]:
                settle_time = x[i] - x[start_i]
                break
        if settle_time < 0:
            print("WARNING: settle time not found")
            settle_time = 0
        settle_times.append(settle_time)
    return settle_times

# piece-wise linear function
def linear_model(x, mn, mp, c):
    y = np.piecewise(x, [x < 0, x >= 0], [lambda x: mn * x + c, lambda x: mp * x + c])
    return y

def analyze_coil(coil, savefig=True):
    dataset_num = dataset_dict[coil]
    data = get_raw_data(dataset_num)
    set_values = get_raw_data(dataset_num, 70+dataset_num)
    set_values = np.array(get_set_values(set_values))

    stab_time = np.array(stabilization_time(data))
    plt.scatter(set_values, stab_time, c='black')
    plt.xlabel("Set values (A)")
    plt.ylabel("Stabilization Time (s)")

    popt, pcov = curve_fit(linear_model, set_values, stab_time) # , sigma=menStd

    x_plot = np.linspace(min(set_values), max(set_values), 1000)
    plt.plot(x_plot, linear_model(x_plot, *popt), \
        label=r'$m_{neg}$=%.2f s/A, $m_{pos}$=%.2f s/A, c=%.2f s'%tuple(popt), c='r')
    plt.legend()
    plt.title(f"{coil} coil settling time")
    if savefig:
        plt.savefig(IMAGEFOLDER+f"{coil}.png")
    plt.close()
    
    return popt

coil = "inj" # {"injection" : 3, "extraction": 4, "middle": 5}

popt_dict = {}
for coil in dataset_dict.keys():
    popt = analyze_coil(coil)
    popt_dict[coil] = popt
    
# dump as json

def cost_model(deltas):
    """Model that predicts settling time (s) of a change

    Args:
        deltas (dict): possible keys 
            'inj', 'ext', 'mid' in units of A
            'baz', in units of ??

    Returns:
        settling time: Unit s
    """
    
    settling_time = 50
    for key in deltas:
        if key in dataset_dict.keys():
            popt = popt_dict[key]
            st = linear_model(deltas[key], *popt)
            settling_time = max(settling_time, st)
            
    return settling_time

deltas = {'inj':-1, 'ext':-1}
print(cost_model(deltas))
    
