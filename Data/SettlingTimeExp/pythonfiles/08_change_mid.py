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

def writeoutput(outfile,readvars,done):
    outfile.write("%.3f "%(venus.read([readvars[0]])))
    for i in range(1,len(readvars)):
        outfile.write("%.5e "%(venus.read([readvars[i]])))
    outfile.write('%1i\n'%(done))

def change_superconductors(Igoal,readvars):
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

        print('\nin change:\nInow=',Inow)
        print('Iaim',Iaim)
        print('Ioff',Ioff)

        checkdone = np.zeros((3,40))+5.0

        start_time = time.time()
        venus.write({'inj_i':Iaim[0], 'ext_i':Iaim[1], 'mid_i':Iaim[2]})
        #print('starting new field setting')

        def check_done(done,Inow,Igoal,Ioff):
            if done[0]==0 and direction[0]*(Inow[0]-Ioff[0])>0:
                venus.write({'inj_i':Igoal[0]}); done[0]=1
                print('inj to goal:',done,' Inow:',Inow[0],' Igoal:',Igoal[0])
            if done[1]==0 and direction[1]*(Inow[1]-Ioff[1])>0:
                venus.write({'ext_i':Igoal[1]}); done[1]=1
                print('ext to goal:',done,' Inow:',Inow[1],' Igoal:',Igoal[1])
            if done[2]==0 and direction[2]*(Inow[2]-Ioff[2])>0:
                venus.write({'mid_i':Igoal[2]}); done[2]=1
                print('mid to goal:',done,' Inow:',Inow[2],' Igoal:',Igoal[2])
            return(done)

        names=['inj_i','ext_i','mid_i']
        diffall = len(checkdone[0,:])*.04
        while np.sum(checkdone[0,:])>diffall or np.sum(checkdone[1,:])>diffall or np.sum(checkdone[2,:])>diffall:
            for i in range(5):
                time.sleep(0.1)
                Inow=np.array([venus.read(['inj_i']),venus.read(['ext_i']),venus.read(['mid_i'])])   # current currents
#                print("%4.0f: %i %6.2f %6.2f %6.2f %6.2f || %i %6.2f %6.2f %6.2f %6.2f || %i %6.2f %6.2f %6.2f %6.2f"%(
#                    time.time()-time_start_change,int(done[0]),Inow[0],Igoal[0],Ioff[0],np.sum(checkdone[0,:])/diffall, 
#                    int(done[1]),Inow[1],Igoal[1],Ioff[1],np.sum(checkdone[1,:])/diffall,
#                    int(done[2]),Inow[2],Igoal[2],Ioff[2],np.sum(checkdone[2,:])/diffall))
                done = check_done(done,Inow,Igoal,Ioff)
                writeoutput(outfile,readvars,0)
                time.sleep(.27)

            if time.time()-time_start_change >300.0:
                print('!!!!!!!!!  timed out !!!!!!!')
                print('trying to set',Igoal)
                print('got stuck at ',venus.read(['inj_i']),venus.read(['ext_i']),venus.read(['mid_i']))   # current currents
                Inow=np.array([venus.read(['inj_i']),venus.read(['ext_i']),venus.read(['mid_i'])])   # current currents
                for i in range(3):
                    print('in here. i=%i, Inow[i]=%f, Igoal[i]=%f, done[i]=%i'%(i,Inow[i],Igoal[i],done[i]))
                    if np.abs(Inow[i]-Igoal[i])<0.08 and done[i]==0:   # for small change problem
                        Igoal[i]=Igoal[i]-.01*np.sign(Inow[i]-Igoal[i])
                        venus.write({names[i]:Igoal[i]})
                        print('stuck with small difference.  resetting Igoal')
                        time_start_change=time.time()
                    if np.abs(Inow[i]-Igoal[i])>=0.08:   # for some reason done going to 1 but not changing I goal
                        done[i]=0
                        venus.write({names[i]:Igoal[i]})
                        print('stuck with big difference. re-requesting Igoal')
                        time_start_change=time.time()
            checkdone[:,:-1] = checkdone[:,1:]; checkdone[:,-1]=np.abs(Inow-Igoal)
        #print('done setting field ')


readvars = venus.read_vars()
coil = 2 #0:inj,1:ext,2:mid
coilname = ['inj','ext','mid']
base = np.array([185.,137.,153.])
currentchange = 3.0
lows = base-currentchange
highs = base+currentchange

for i in range(25):
    tstart = time.time()
    fname = 'data/'+coilname[coil]+'_'+str(int(tstart))
    outfile = open(fname,'w')
    Igoal = base*1.0
    Igoal[coil] = np.random.uniform(low = lows[coil], high = highs[coil])
    for j in range(50):
            writeoutput(outfile,readvars,0)
            time.sleep(0.37)
    change_superconductors(Igoal,readvars)
    for j in range(750):
            writeoutput(outfile,readvars,1)
            time.sleep(0.37)
    outfile.close()
    print(coilname[coil],' file ',i,' took ',time.time()-tstart)
change_superconductors(base,readvars)
time.sleep(60)
