# -*- coding: utf-8 -*-
#getrefv.py

from __future__ import print_function, division
import numpy as np

class GetRefV(object):

    def __init__(self, params):
        super(GetRefV, self).__init__()
        
        self.params = params
        self.ReadVoltage = params.ReadVoltage
        self.numCol = params.numCol
        self.numRow = params.numRow
        self.WeightBits = params.WeightBits
        self.CellBits = params.CellBits
        self.IOBits = params.IOBits
        self.numCellperWeight = params.numCellperWeight


    def apply(self, Input, WeightArrays, numLayerOutput, LayerInfo="Linear",
            input_size_x=0, input_size_y=0, kernel_size_x=0, kernel_size_y=0):
        """
        Find the max current as AD reference current among the cores in a layer
        require args:
            Input: The Inputs of a layer
            WeightArrays: the compiled weights of a layer
            numLayerOutput: the output size of a layer
            LayerInfo: "Linear" or "Conv"
        """
        if LayerInfo == "Linear":
            MaxCurrent = 0
            for i in range(len(Input)):
                Sample = Input[i]
                for j in range(len(WeightArrays)):
                    InputCore = Sample[j]
                    WeightCoreH = WeightArrays[j]
                    numOutputRemain = numLayerOutput * self.numCellperWeight
                    for k in range(len(WeightCoreH)):
                        WeightCore = WeightCoreH[k]
                        for l in range(self.IOBits):# IOBits
                            OutputCore = np.dot(np.transpose(InputCore[l]),
                                WeightCore) * self.ReadVoltage
                            if numOutputRemain > self.numCol:
                                OutputCore = OutputCore
                            else:
                                
                                OutputCore = OutputCore[0:numOutputRemain]
                            MaxCurrent = max(MaxCurrent, OutputCore.max())
                        numOutputRemain = numOutputRemain - \
                            (self.numCol - self.numCellperWeight)
            return MaxCurrent

