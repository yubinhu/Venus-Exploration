"""
This module rewrites cost.py to include cost_fn_pure from simulator.py inside the class CostModel
and rename it to CostModel.currents_to_cost()

It should load the POPT_DICT from "Models/costmodel.json"
It should have the function
"""
import numpy as np
import json
with open("Models/costmodel.json", "r") as f:
    POPT_DICT = json.load(f) # default cost model

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
    
    @staticmethod
    def currents_to_cost(present_currents, new_currents):
        # return the time to set currents in seconds. Use default cost model. 
        # present_currents = [inj, mid, ext]
        if len(new_currents.shape) > 1:
            # two dimensional array
            costs = []
            for i in range(len(new_currents)):
                deltas = {'inj': new_currents[i][0] - present_currents[0], 
                    'mid': new_currents[i][1] - present_currents[1], 
                    'ext': new_currents[i][2] - present_currents[2]}
                costs.append(CostModel(POPT_DICT)(deltas))
            return np.array(costs)
        else:
            deltas = {'inj': new_currents[0] - present_currents[0], 
                'mid': new_currents[1] - present_currents[1], 
                'ext': new_currents[2] - present_currents[2]}
            return CostModel(POPT_DICT)(deltas)