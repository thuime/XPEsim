# -*- coding: utf-8 -*-
#layercal.py

from __future__ import print_function, division
import numpy as np
from .corecal import CoreCal

class LayerCal(object):

    def __init__(self, params):
        super(LayerCal, self).__init__()
        
        self.params = params
        self.ReadVoltage = params.ReadVoltage
        self.numCol = params.numCol
        self.numRow = params.numRow
        self.WeightBits = params.WeightBits
        self.CellBits = params.CellBits
        self.IOBits = params.IOBits
        self.ADBits = params.ADBits
        self.numCellperWeight = params.numCellperWeight
        
        self.corecal = CoreCal(params)

    def DenseCal(self, InputBatch, WeightArray, MaxCurrent, numLayerOutput):
        """
        Calculate outputs in dense layer
        require args:
            InputBatch: The Inputs of a layer
            WeightArray: the compiled weights of a layer
            MaxCurrent: the reference current of a layer
        """
        Output = []
        numCoreV = len(WeightArray)
        numCoreH = len(WeightArray[0])
        for i in range(len(InputBatch)):
            Sample = InputBatch[i]
            OutputPerSample = np.zeros((self.numCol//self.numCellperWeight - 1) *
                    numCoreH)
            for j in range(numCoreV):
                InputCoreV = Sample[j]
                WeightCoreV = WeightArray[j]
                OutputCoreV = np.asarray([])
                for k in range(numCoreH):
                    WeightCore = WeightCoreV[k]
                    OutputCore = self.corecal.apply(InputCoreV, WeightCore, MaxCurrent)
                    OutputCoreV = np.concatenate((OutputCoreV, OutputCore), axis=0)
                OutputPerSample += OutputCoreV
            Output.append(OutputPerSample)
        return np.asarray(Output)[:,:numLayerOutput]
