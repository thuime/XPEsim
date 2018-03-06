# -*- coding: utf-8 -*-
#activationfunc.py

from __future__ import print_function, division
import numpy as np

class ActivationFunc(object):

    def __init__(self, params):
        super(ActivationFunc, self).__init__()
        self.params = params
        self.IOBits = params.IOBits

    def apply(self, Input, Func):
        if Func == "ReLU" or Func == "relu":
            Output = np.where(Input > 0, Input, 0)
            
            # 8 bits output to next layer
            MaxBits = int(np.ceil(np.log2(Output.max())))
            LSB = MaxBits - self.IOBits
            MSB = MaxBits
#==============================================================================
#             LSB = 18
#             MSB = LSB + self.IOBits
#==============================================================================     
            Output = Output % ( 2**MSB) // ( 2**LSB)
        return Output
