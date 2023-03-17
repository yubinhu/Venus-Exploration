
class VenusController:
    # Wrapper class for controlling the venusplc
    
    def __init__(
        self,
        inj_limits=[182, 188],
        ext_limits=[134, 140],
        mid_limits=[149, 155],
    ):
        """The limits on the magnetic solenoids currents and the beam range (ouput).
        A random jitter can be added also (fraction of 1.)."""
        self.inj_limits = inj_limits
        self.ext_limits = ext_limits
        self.mid_limits = mid_limits
        self.currents = np.zeros(3)
        self.rng = np.random.default_rng(42)

        self.device = venusplc.VENUSController(read_only=False)
        self.logs = []

    def change_superconductors(self, Igoal):
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
        for v, lim in zip([inj, ext, mid], [self.inj_limits, self.ext_limits, self.mid_limits]):
            if v < lim[0] or v > lim[1]:
                raise ValueError("Setting outside limits")
        # self.currents = np.array([inj, mid, ext])
        venus = self.device

        if 0: # dst: hiding these and using the faster change Superconductors program
            venus.write({'inj_i':inj})         # injection solenoid current [A]
            venus.write({'ext_i':ext})         # extraction solenoid current [A]
            venus.write({'mid_i':mid})         # middle solenoid current [A]
        self.change_superconductors(np.array([inj,ext,mid]))

    def change_something(self, thing, value):
        venus = self.device
        venus.write({thing:value})

    def read_something(self, thing):
        venus = self.device
        return(venus.read([thing]))

    def get_noise_level(self):
        # return std of the noise
        # noise = self.jitter*(self.beam_range[1] - self.beam_range[0])
        return 0.1 # assume 0.1% noise
    
    def get_bounds(self, i):
        if i == 0:
            return self.inj_limits
        elif i==1:
            return self.ext_limits
        elif i==2:
            return self.mid_limits
    
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
