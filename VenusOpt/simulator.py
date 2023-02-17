import numpy as np
from VenusOpt.cost import CostModel
class Venus:
    # This is a Venus simulator
    # note: always set the func paramter
    def __init__(
        self,
        inj_limits=[116, 128],
        mid_limits=[97, 110],
        ext_limits=[97, 110],
        beam_range=[0.50, 1.00],
        jitter=0.0, 
        func = (lambda X: -1)
    ):
        """The limits on the magnetic solenoids currents and the beam range (ouput).
        A random jitter can be added also (fraction of 1.)."""
        self.inj_limits = inj_limits
        self.mid_limits = mid_limits
        self.ext_limits = ext_limits
        self.beam_range = beam_range
        self.currents = np.zeros(3)
        self.jitter = jitter
        self.rng = np.random.default_rng(42)
        self.func = func
        self.time_cost = 0
    
    @staticmethod
    def cost_fn_pure(present_currents, new_currents):
        # return the time to set currents in seconds
        # Time model subject to change. Wait on some inputs from Damon. 
        # Bias change in a second, coils change in minutes
        # Proposed model: max_i ( control_time_i )
        # control_time_i = const_i * delta_i + plasma_term_i(delta_i)
        # plasma_term_i is const or zero (depend on tolerance)
        deltas = {'inj': new_currents[0] - present_currents[0], 
            'mid': new_currents[1] - present_currents[1], 
            'ext': new_currents[2] - present_currents[2]}
        return CostModel('Models/cost_model.json')(deltas)
    
    def _cost_fn(self, present_currents, new_currents):
        return self.cost_fn_pure(present_currents, new_currents) + self.rng.normal(0.0, self.jitter)

    def get_total_time(self):
        return self.time_cost
    
    def set_mag_currents(self, inj, mid, ext):
        """Set the magnetic currents on the coils."""
        for v, lim in zip([inj, mid, ext], [self.inj_limits, self.mid_limits, self.ext_limits]):
            if v < lim[0] or v > lim[1]:
                raise ValueError("Setting outside limits")
        
        new_currents = np.array([inj, mid, ext])
        
        # calculate the time cost to set currents
        self.time_cost += self._cost_fn(self.currents, new_currents)
        
        self.currents = new_currents

    def _rescale_inputs(self, inputs):
        """input to himmelblau4 must be in [-6, 6]."""
        return (
            (c - l[0]) * 12.0 / (l[1] - l[0]) - 6.0
            for c, l in zip(inputs, [self.inj_limits, self.mid_limits, self.ext_limits])
        )

    def _rescale_output(self, output):
        """simple square returns values betwen 0 and 27 for w, x, y, z in [-6, 6]."""
        return (
            (1. - (output / 27.0) + self.rng.normal(0.0, self.jitter)) *
            (self.beam_range[1] - self.beam_range[0]) + self.beam_range[0]
        )
    
    def get_noise_level(self):
        # return std of the noise
        noise = self.jitter*(self.beam_range[1] - self.beam_range[0])
        return noise

    def get_beam_current(self):
        """Read the current value of the beam current"""
        self.time_cost += 60 # it takes 60 seconds to read the beam current
        return self.func(self.currents.reshape(1, -1)) + self.rng.normal(0.0, self.jitter)

    def bbf(self, A, B, C):
        self.set_mag_currents(A, B, C)
        v = self.get_beam_current()
        return v
    
    def bbf_named(self, inj_i_mean, mid_i_mean, ext_i_mean):
        self.set_mag_currents(inj_i_mean, mid_i_mean, ext_i_mean)
        v = self.get_beam_current()
        return v

    @staticmethod
    def _simple_square(w, x, y):
        """A not so funky 3 dimensional parameter space with a single minima."""
        return (
            (w - 3.)**2 + (x - 3.)**2 + (y - 3.)**2
        )