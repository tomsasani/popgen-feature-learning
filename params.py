"""
Utility functions and classes (including default parameters).
Author: Sara Mathieson, Rebecca Riley
Date: 9/27/22
"""

# python imports
import numpy as np
import sys


class Parameter:
    """
    Holds information about evolutionary parameters to infer.
    Note: the value arg is NOT the starting value, just used as a default if
    that parameter is not inferred, or the truth when training data is simulated
    """

    def __init__(self, value, min, max, name):
        self.value = value
        self.min = min
        self.max = max
        self.name = name
        self.proposal_width = (self.max - self.min) / 1

    def initialize(self, rng: np.random.default_rng):
        return rng.uniform(self.min, self.max)

    def fit_to_range(self, value):
        value = min(value, self.max)
        return max(value, self.min)

    def proposal(self, cur_val, temperature, rng):
        """propose a new value for the parameter of interest.
        we make the range of possible parameter choices smaller
        as time goes on in order to encourage hard negative mining.
        the range is controlled by the temperature. large temperatures
        (at the beginning of training) permit wider proposal widths.

        Args:
            temperature (float): temperature value that controls width
                of possible proposals

        Returns:
            _type_: _description_
        """
        if temperature <= 0: # last iter
            return cur_val

        # normal around current value (make sure we don't go outside bounds)
        new_val = rng.normal(cur_val, self.proposal_width * temperature)
        new_val = self.fit_to_range(new_val)
        # if the parameter hits the min or max it tends to get stuck
        if new_val == cur_val or new_val == self.min or new_val == self.max:
            return self.proposal(cur_val, temperature, rng) # recurse
        else:
            return new_val

class ParamSet:

    def __init__(self):

        # population sizes and bottleneck times
        self.N1 = Parameter(9000, 1000, 30000, "N1")
        self.N2 = Parameter(5000, 1000, 30000, "N2")
        self.T1 = Parameter(2000, 1500, 5000, "T1")
        self.T2 = Parameter(350, 100, 1500, "T2")

        self.rho = Parameter(1e-8, 1e-9, 3e-8, "rho")
        self.mu = Parameter(1e-8, 1e-9, 3e-8, "mu")
        self.growth = Parameter(0.005, 0.0, 0.05, "growth")


    def update(self, names, values):
        """Based on generator proposal, update desired param values"""
        assert len(names) == len(values)

        for j in range(len(names)):
            param = names[j]
            # credit: Alex Pan (https://github.com/apanana/pg-gan)
            attr = getattr(self, param)
            if attr is None:
                sys.exit(param + " is not a recognized parameter.")
            else:
                attr.value = values[j]


if __name__ == "__main__":

    rng = np.random.default_rng(42)

    parameters = ParamSet()

    for i in range(10):
        print (parameters.N2.proposal(9_000, 1, rng))

    prop = parameters.N1.proposal(9_000, 1, rng)
