"""
This module rewrites cost.py to include cost_fn_pure from simulator.py inside the class CostModel
and rename it to CostModel.currents_to_cost()

It should load the POPT_DICT from "Models/costmodel.json"
It should have the function
"""
from typing import List
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
    
    def build_cost_function(self, parameters:List[str]):
        """Build a static cost function for the given parameters
        Args:
            parameters (List[str]): List of parameters to include in the cost function. 
                Possible values are 'inj', 'ext', 'mid', 'baz1', 'baz2', 'bias'.
                Order is important, as it determines the meaning of each entries in the
                cost function input array.
        Returns:
            cost_function: A static function that takes a numpy array and returns 
                a cost according to the cost model
        """
        def cost_function(old, new):
            # assert len(old) == len(new)
            # assert len(old) == len(parameters)
            if len(old.shape) > 1:
                res = []
                for i in range(len(old)):
                    res.append(cost_function(old[i], new[i]))
                return np.array(res)
            else:
                d = {}
                for i in range(len(old)):
                    d[parameters[i]] = new[i] - old[i]
                return self(d)
        return cost_function
    
    # piece-wise linear function
    @staticmethod
    def linear_model(x, mn, mp, c):
        y = np.piecewise(x, [x < 0, x >= 0], [lambda x: mn * x + c, lambda x: mp * x + c])
        return y