#!/bin/env python3

# Note: run this script from the root directory of this project so that costmodel loads properly. (Harvey)

import inspect
import itertools
import time
import types
import signal
import numpy as np
import datetime
import time
import statistics
from VenusOpt.cost import CostModel
import json


from bayes_opt import BayesianOptimization, UtilityFunction
import warnings
warnings.filterwarnings("ignore")

import venus_data_utils.venusplc as venusplc

#import plcauto.dbwriter as dbwriter
#import plcauto.maths as maths

class VenusController:
    """Wrapper class for controlling the venusplc.
        This class exisits to:
            1. Implement higher level control logic required for tuning Venus
            2. Provide gardrails on accessing the hardware so we don't have to trust the BayesOpt library
            3. Bridge the gap between hardware experiment and software simulation
    """
    
    def __init__(
        self,
        pbounds = {}
    ):
        # pbounds sets the limits on the parameters we are adjusting. This guardrail is in place 
        # in case we do not trust the BayesOpt library to stay within the limits
        
        # These bounds will used if no pbounds are provided for a parameter
        default_pbounds = {"inj": (182, 188), "ext": (134, 140), "mid": (149, 155), 
                           "gas_balzer_2": (12.5,13.5), "bias_v":(25.0,80.0)} 
        #TODO(Harvey): check if we want to change the name of the currents
        
        for key in default_pbounds.keys():
            if key not in pbounds:
                pbounds[key] = default_pbounds[key]
                
        self.pbounds = pbounds
        self.device = venusplc.VENUSController(read_only=False)
        self.logs = []

    def change_superconductors(self, Igoal):
        # TODO: check Igoal is within limits
        venus = self.device
        time_start_change = time.time()
        usefastdiff = 0.1  # only use the fast search if the difference between current and goal is > this amount
        Inow=np.array([venus.read(['inj_i']),venus.read(['ext_i']),venus.read(['mid_i'])])   # current currents

        done = np.zeros(3)+1                    # int array to check if search is done
        direction = np.sign(Igoal-Inow)         # di
        Idiff = np.abs(Igoal-Inow)
        done[np.where(Idiff>usefastdiff)]=0

        Iaim = np.zeros(3)
        Iaim[np.where(direction>0)]=Igoal[np.where(direction>0)]+5
        Iaim[np.where(direction<0)]=Igoal[np.where(direction<0)]-5
        Iaim[np.where(done==1)]=Igoal[np.where(done==1)]


        diffup = np.array([.03,.04,.08])
        diffdown = np.array([.06,.10,.25])
        Ioff = Igoal*1.0
        for i in range(len(Ioff)):
            if direction[i]>0: Ioff[i]=Ioff[i]-direction[i]*diffup[i]
            if direction[i]<0: Ioff[i]=Ioff[i]-direction[i]*diffdown[i]


        checkdone = np.zeros((3,40))+5.0

        start_time = time.time()
        venus.write({'inj_i':Iaim[0], 'ext_i':Iaim[1], 'mid_i':Iaim[2]})
        #print('starting new field setting')

        def check_done(done,Inow,Igoal,Ioff):
            if done[0]==0 and direction[0]*(Inow[0]-Ioff[0])>0:
                venus.write({'inj_i':Igoal[0]}); done[0]=1
            if done[1]==0 and direction[1]*(Inow[1]-Ioff[1])>0:
                venus.write({'ext_i':Igoal[1]}); done[1]=1
            if done[2]==0 and direction[2]*(Inow[2]-Ioff[2])>0:
                venus.write({'mid_i':Igoal[2]}); done[2]=1
            return(done)

        names=['inj_i','ext_i','mid_i']
        diffall = len(checkdone[0,:])*.04
        while np.sum(checkdone[0,:])>diffall or np.sum(checkdone[1,:])>diffall or np.sum(checkdone[2,:])>diffall:
            for i in range(5):
                time.sleep(0.1)
                Inow=np.array([venus.read(['inj_i']),venus.read(['ext_i']),venus.read(['mid_i'])])   # current currents
                done = check_done(done,Inow,Igoal,Ioff)

            if time.time()-time_start_change >300.0:
                Inow=np.array([venus.read(['inj_i']),venus.read(['ext_i']),venus.read(['mid_i'])])   # current currents
                for i in range(3):
                    if np.abs(Inow[i]-Igoal[i])<0.08 and done[i]==0:   # for small change problem
                        Igoal[i]=Igoal[i]-.01*np.sign(Inow[i]-Igoal[i])
                        venus.write({names[i]:Igoal[i]})
                        time_start_change=time.time()
                    if np.abs(Inow[i]-Igoal[i])>=0.08:   # for some reason done going to 1 but not changing I goal
                        done[i]=0
                        venus.write({names[i]:Igoal[i]})
                        time_start_change=time.time()
            checkdone[:,:-1] = checkdone[:,1:]; checkdone[:,-1]=np.abs(Inow-Igoal)

    def set_mag_currents(self, inj, ext, mid):
        """Set the magnetic currents on the coils."""
        self.change_superconductors(np.array([inj,ext,mid]))

    def change_something(self, thing, value):
        if thing in self.pbounds and (value < self.pbounds[thing][0] or value > self.pbounds[thing][1]):
            raise ValueError("Setting outside limits. Trying to set {} to {}".format(thing, value))
        self.device.write({thing:value})

    def read_something(self, thing):
        venus = self.device
        return(venus.read([thing]))

    def get_noise_level(self):
        # return std of the noise
        # noise = self.jitter*(self.beam_range[1] - self.beam_range[0])
        return 0.1 # assume 0.1% noise
    
    def get_beam_current(self):
        """Read the current value of the beam current"""
        venus = self.device
        Ifc = venus.read(['fcv1_i'])*1e6    # faraday cup current (single species current) in microamps
        return Ifc
    
    def dump_log(self, logfile="log.json"):
        json.dump(self.logs, open(logfile, "w"))

    def get_readvars(self):
        venus = self.device
        return(venus.read_vars())

    def monitor(self,t_start,t_program_start,output_file,readvars,output_full):
        venus = self.device
        Ifc = venus.read(['fcv1_i'])*1e6        # faraday cup current (single species current) in microamps
        Idrain = venus.read(['extraction_i'])   # drain current ~ total extracted beam current [mA]
        Pinj = venus.read(['inj_mbar'])         # injection pressure [torr]

        # things also worth monitoring to understand system
        Ibias = venus.read(['bias_i'])          # bias disk current [mA]
        Ipull = venus.read(['puller_i'])        # puller electrode current [mA]
        Xsrc = venus.read(['x_ray_source'])     # amount of x-rays produced by source [?]
        Pext = venus.read(['ext_mbar'])         # pressure just outside source [torr]
        HHe = venus.read(['four_k_heater_power'])   # liquid He heater power [W]

        # TODO: write to database or somewhere
        output_full.write("%.3f "%(venus.read([readvars[0]])))
        for i in range(1,69):
            output_full.write("%.5e "%(venus.read([readvars[i]])))
        output_full.write("\n")
        output_file.write("%7.1f %12.3f %12.3f %10.4f %10.4f %8.2e %8.2e %7.2f %7.2f %7.2f %7.2f %7.1f %7.2f %7.2f\n"%(time.time()-t_start,
            time.time()-t_program_start,venus.read(['fcv1_i'])*1e6,venus.read(['extraction_i']),venus.read(['bias_i']),
            venus.read(['inj_mbar']),venus.read(['ext_mbar']),venus.read(['inj_i']),venus.read(['ext_i']),venus.read(['mid_i']),
            venus.read(['sext_i']),venus.read(['x_ray_source']),venus.read(['bias_v']),venus.read(['gas_balzer_2'])))
        #time.sleep(1)

pbounds = {"mid": (149, 155), "gas_balzer_2": (12.5,13.5), "bias_v":(25.0,80.0)}
porder = ["mid", "gas_balzer_2", "bias_v"] # order of the parameters for black_box_function

venus = VenusController(pbounds)

basedir = '/home/damon/Venus-Exploration/Scripts/20230306_bayes_cost_weighting/'
writefile = open(basedir+'data/monitor_harvey_'+str(int(time.time())),'w')
readvars = venus.get_readvars()
writefilefull = open(basedir+'data/monitor_full_'+str(int(time.time())),'w')
t_program_start = time.time()

# defining a cost function. Input rel_std. Output is in scale to std of mean of beam currrent
squared = lambda x: 0.5*(20*x)**2
BEAM_CURR_STD = 30

# Define the black box function to optimize.
#def black_box_function(inj, mid, ext):  #Harvey
def black_box_function(mid, gas_balzer_2, bias_v):  #Harvey
    mid0 = venus.read_something('mid_i')
    balzer0 = venus.read_something('gas_balzer_2')
    bias0 = venus.read_something('bias_v')
    print(f'going from {mid0:.2f} to {mid:.2f} | {balzer0:.2f} to {gas_balzer_2:.2f} | {bias0:.2f} to {bias_v:.2f}')
    venus.change_something('bias_v',bias_v)
    venus.change_something('gas_balzer_2',12.0)
    while(venus.read_something('gas_balzer_2')>12.1):
        time.sleep(0.37)
    venus.change_something('gas_balzer_2',gas_balzer_2)
    venus.set_mag_currents(185, 137, mid)
    t_start = time.time()
    t_end = time.time() + 1 * 60 # wait for 5min   #dst 1 min
    #print('monitoring...')
    while time.time() < t_end:
        venus.monitor(t_start,t_program_start,writefile,readvars,writefilefull)
        time.sleep(0.37)

    # dst t_end = time.time() + 10 # data acquisition for 10s
    v_list = []
    # dst while time.time() < t_end:
    for i in range(25):
        v = venus.get_beam_current()
        v_list.append(v)
        time.sleep(.37)
    v_mean = sum(v_list) / len(v_list)
    v_std = statistics.stdev(v_list)
    rel_std = v_std / v_mean
    instability_cost = squared(rel_std) * BEAM_CURR_STD
    # save the v list?
    print(f'average and rel_std*100 current for 25 measurements: {v_mean:.2f} {rel_std*100.:.2f}')
    return v_mean - instability_cost

def denormalize(pbounds, porder, x):
    assert len(x) == len(porder)
    out = []
    for i in range(len(x)):
        out.append(pbounds[porder[i]][0] + x[i] * (pbounds[porder[i]][1] - pbounds[porder[i]][0]))
    return out

def black_box_function_normalized(mid_norm, gas_balzer_2_norm, bias_v_norm):
    mid, gas_balzer_2, bias_v = denormalize(pbounds, porder, [mid_norm, gas_balzer_2_norm, bias_v_norm])
    return black_box_function(mid, gas_balzer_2, bias_v)
normed_pbounds = {'mid': (0, 1), 'gas_balzer_2': (0, 1), 'bias_v': (0, 1)}

with open("Models/costmodel.json", "r") as f:
    POPT_DICT = json.load(f)
cost_function = CostModel(popt_dict=POPT_DICT).build_cost_function(porder)

optimizer = BayesianOptimization(f = black_box_function, pbounds = normed_pbounds, verbose = 0, 
                                 allow_duplicate_points=True)

acq_func = UtilityFunction(kind='eipu', kappa=2.5, xi=0.01, kappa_decay=1, kappa_decay_delay=0, cost_func=cost_function)
optimizer.set_gp_params(alpha=0.15, kernel__length_scale=10, kernel__nu=0.5)

noise = venus.get_noise_level()
optimizer.maximize(init_points = 5, n_iter = 30, acquisition_function=acq_func) #
print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))

