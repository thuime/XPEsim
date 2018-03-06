# -*- coding: utf-8 -*-
#layercal.py

from __future__ import print_function, division
import numpy as np

class LayerCal(object):

    def __init__(self, params):
        super(LayerCal, self).__init__()
        self.params = params
        self.ReadVoltage = float(self.params["ReadVoltage"])
        self.numCol = int(self.params["numArrayCol"])
        self.numRow = int(self.params["numArrayRow"])
        self.WeightBits = int(self.params["WeightBits"])
        self.CellBits = int(self.params["CellBits"])
        self.IOBits = int(self.params["IOBits"])
        self.ADBits = 8
        self.numCellperWeight = int(np.ceil(self.WeightBits/self.CellBits))

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
        numOutputPerCore = self.numCol//self.numCellperWeight-1
        Base = np.asarray([(2**self.CellBits)**x 
                            for x in range(self.numCellperWeight-1, -1, -1)])
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
                    OutputCore = np.zeros(numOutputPerCore)
                    for l in range(self.IOBits):
                        # Calculate output in Core
                        OutputPerBit = np.dot(np.transpose(InputCoreV[l]),
                                WeightCore) * self.ReadVoltage
                        # ADC Output
                        OutputPerBit = np.round(OutputPerBit / MaxCurrent *
                                (2 ** self.ADBits - 1))
                        OutputPerBit = OutputPerBit[0]
                        # Sum Output
                        OutputSum = np.zeros(numOutputPerCore+1)
                        for m in range(self.numCellperWeight):
                            OutputSum += OutputPerBit[m:self.numCol:
                                self.numCellperWeight] * Base[m]
                        OutputSum = OutputSum[0:-1] - OutputSum[-1]
                            #OutputPerBit[-self.numCellperWeight] * (Base[0]+1)
                        #Shift Adder Output
                        OutputCore += OutputSum * (2 ** (self.IOBits - l))
                    OutputCoreV = np.concatenate((OutputCoreV, OutputCore), axis=0)
                OutputPerSample += OutputCoreV
            Output.append(OutputPerSample)
        return np.asarray(Output)[:,:numLayerOutput]
