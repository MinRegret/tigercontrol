# Problem class
# Author: John Hallman

from ctsb import error
from ctsb.problems import Problem

# class for online control tests
class TimeSeriesProblem(Problem):

    def initialize(self, **kwargs):
        # resets problem to time 0
        self.has_regressors = None
        raise NotImplementedError

    def step(self, action=None):
        #Run one timestep of the problem's dynamics. 
        raise NotImplementedError

