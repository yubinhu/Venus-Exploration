# This file is for the cost model
import numpy as np

class CostModel:
    def __init__(self, popt_dict):
        self.popt_dict = popt_dict
        
    def __call__(self, deltas):
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
            if key in self.popt_dict.keys():
                popt = self.popt_dict[key]
                st = self.linear_model(deltas[key], *popt)
                settling_time = max(settling_time, st)
        return settling_time
    
    # piece-wise linear function
    @staticmethod
    def linear_model(x, mn, mp, c):
        y = np.piecewise(x, [x < 0, x >= 0], [lambda x: mn * x + c, lambda x: mp * x + c])
        return y