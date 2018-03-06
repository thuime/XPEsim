# -*- coding: utf-8 -*-
#corecal.py

from __future__ import print_function, division
import numpy as np

class CoreCal(object):

    def __init__(self, params):
        super(CoreCal, self).__init__()
        
        self.params = params
        self.ReadVoltage = params.ReadVoltage
        self.numCol = params.numCol
        self.numRow = params.numRow
        self.WeightBits = params.WeightBits
        self.CellBits = params.CellBits
        self.IOBits = params.IOBits
        self.ADBits = params.ADBits
        
        self.numCellperWeight = int(np.ceil(self.WeightBits/self.CellBits))
        self.numOutputPerCore = self.numCol//self.numCellperWeight-1
        self.Base = np.asarray([(2**self.CellBits)**x 
                            for x in range(self.numCellperWeight-1, -1, -1)])

    def apply(self, Input, WeightCore, MaxCurrent):
        """
        Calculate outputs in core
        require args:
            Input: The Inputs of a core
            WeightCore: the compiled weights of core
            MaxCurrent: the reference current of a layer
        """
        OutputCore = np.zeros(self.numOutputPerCore)
        for l in range(self.IOBits):
            # Calculate output in Core
            OutputPerBit = np.dot(np.transpose(Input[l]),
                    WeightCore) * self.ReadVoltage
            # ADC Output
            OutputPerBit = np.round(OutputPerBit / MaxCurrent *
                    (2 ** self.ADBits - 1))
            OutputPerBit = OutputPerBit[0]
            # Sum Output
            if self.numCellperWeight > 1:
                OutputSum = np.zeros(self.numOutputPerCore+1)
                for m in range(self.numCellperWeight):
                    OutputSum += OutputPerBit[m:self.numCol:
                            self.numCellperWeight] * self.Base[m]
                OutputSum = OutputSum[0:-1] - OutputSum[-1]
               #OutputPerBit[-self.numCellperWeight] * (Base[0]+1)
            #Shift Adder Output
                OutputCore += OutputSum * (2 ** (self.IOBits - l))
            else:
                OutputSum = OutputPerBit
                OutputSum = OutputSum[0:-1] - OutputSum[-1]
                OutputCore += OutputSum * (2 ** (self.IOBits - l))
        return OutputCore
