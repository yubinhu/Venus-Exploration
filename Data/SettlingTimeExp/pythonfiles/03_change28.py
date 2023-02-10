#!/bin/env python3

import inspect
import itertools
import time
import types
import signal
import numpy as np
import datetime
import time
import statistics


import venus_data_utils.venusplc as venusplc
        
#venus = venusplc.VENUSController('/home/damon/venus_data_project/config/config.ini')
venus = venusplc.VENUSController(read_only=False)

def writeoutput(outfile,readvars):
    outfile.write("%.3f "%(venus.read([readvars[0]])))
    for i in range(1,len(readvars)):
        outfile.write("%.5e "%(venus.read([readvars[i]])))
    outfile.write("\n")

readvars = venus.read_vars()

for i in range(25):
    tstart = time.time()
    fname = 'data/p28_'+str(int(tstart))
    outfile = open(fname,'w')
    powervalue = np.random.uniform(low = 2000., high = 4300.0)
    for j in range(50):
            writeoutput(outfile,readvars)
            time.sleep(0.37)
    venus.write({'g28_req':powervalue})
    for j in range(750):
            writeoutput(outfile,readvars)
            time.sleep(0.37)
    outfile.close()
    print('28 file ',i,' took ',time.time()-tstart)
venus.write({'g28_req':4300.0})
time.sleep(60)
