#SystemCal.py

from __future__ import print_function, division
import numpy as np
import csv
import os
from sysfunctional import AccCal 
from module import NNcompile, Inputcompile, GetRefV
from module import ActivationFunc, LayerCal
from module import HWEval

class SystemSim(object):
    def __init__(self,
            params):
        super(SystemSim, self).__init__()
        self.params = params
        self.Gmax = params.Gmax
        self.Gmin = params.Gmin
        self.ReadVoltage = params.ReadVoltage
        self.numCol = params.numCol
        self.numRow = params.numRow
        self.WeightBits = params.WeightBits
        self.CellBits = params.CellBits
        self.IOBits = params.IOBits
        self.ADBits = params.ADBits
        self.numCoreVMax = params.numCoreVMax
        self.numCoreHMax = params.numCoreHMax
        self.ReadPulseWidth = params.ReadPulseWidth
        self.numCellperWeight = params.numCellperWeight
        self.numLayerOutput = params.numLayerOutput
        self.RangeMax = params.RangeMax # The range of bitwise weight is from -self.RangeMax to (self.RangeMax-1)
        
        self.Accuracy = 0
        self.CoresInfo = [] # numCoreV and numCoreH in every layer
        self.numCore = 0

        self.HWEval_ = HWEval();
    
    def apply(self,
            Net,
            Weight,
            Input,
            Label=None):
        """
        Numerical computing of hardware
        """
        # Linear and Conv
        Linear = []
        self.numLinear = 0
        Conv = []
        self.numConv = 0
        for i in Net:
            if i[0] == "NoiseConv2d" or i[0] == "Conv2d":
                weight = Weight[i[0]+str(self.numConv)]["kernel"]
                bias = Weight[i[0]+str(self.numConv)]["bias"]
                Conv.append((weight, bias))
                self.numConv += 1
            if i[0] == "NoiseLinear" or i[0] == "Linear":
                weight = Weight[i[0]+str(self.numLinear)]["weight"]
                bias = Weight[i[0]+str(self.numLinear)]["bias"]
                Linear.append((weight, bias))
                self.numLinear += 1

        # pooling parameter
        poolHeight, poolWidth = 2, 2
        stride_y, stride_x = 2, 2

        # input data pretreatment
        Input = np.array(list(Input), copy=False)
        Input = np.round(Input * (2 ** self.IOBits - 1))
        Input = Input.reshape(Input.shape[0], 1, 28, 28)
        
        self.BatchSize = len(Input)
        self.WeightArrays = []
        self.InputArray = []
        self.MaxCurrent = []
        
        nncompiler = NNcompile(self.params)
        inputcompiler = Inputcompile(self.params)
        getrefv = GetRefV(self.params)
        activationfunc = ActivationFunc(self.params)
        layercal = LayerCal(self.params)

        
        for i in range(self.numConv):
            # Mapping Conv to array
            print("Start to compile conv layer-%d" % (i+1))
            ConvKernel, bias = Conv[i]
            height, width = ConvKernel.shape[2:4]
            im_height, im_width = Input.shape[2:4]
            
            ConvKernel = ConvKernel[:,:,::-1,::-1].reshape(ConvKernel.shape[0], -1).T
            weight = np.concatenate((ConvKernel, bias.reshape(1, -1)), axis=0)
            WeightArrays_, CoresInfo_ = nncompiler.apply(weight)
            
            self.WeightArrays.append(WeightArrays_)
            self.CoresInfo.append(CoresInfo_)

            # Conv Calculation
            print("Start to calculate conv layer-%d" % (i+1))
            im_2d = []
            for y in range(im_height - height + 1):
                im_1d = []
                for x in range(im_width - width + 1):
                    recept_field = Input[:, :, y:y+height, x:x+width].reshape(Input.shape[0], -1)
                    InputPulse = inputcompiler.apply(recept_field)
                    self.InputArray.append(InputPulse)
                    maxCurrent = getrefv.apply(InputPulse, WeightArrays_, weight.shape[1])
                    self.MaxCurrent.append(maxCurrent)
                    pixOutput = layercal.DenseCal(InputPulse, WeightArrays_, maxCurrent, weight.shape[1])
                    im_1d.append(pixOutput)
                im_2d.append(im_1d)
            Input = np.asarray(im_2d)
            Input = activationfunc.apply(Input, "ReLU")

            # Max pool
            im_height, im_width = Input.shape[0:2]
            im_2d = []
            for y in range(0, im_height, stride_y):
                im_1d = []
                for x in range(0, im_width, stride_x):
                    tmp = Input[y:y+poolHeight, x:x+poolWidth, :, :].max(axis=(0, 1))
                    im_1d.append(tmp)
                im_2d.append(im_1d)
            Input = np.asarray(im_2d)

            Input = np.transpose(Input, (2, 3, 0, 1))

        # Flatten input
        Input = Input.reshape(Input.shape[0], -1)

        # FC layer
        for i in range(self.numLinear):
            print("Start to compile fc layer-%d" % (i+1))
            weight, bias = Linear[i]
            weight = np.concatenate((weight, bias.reshape(1, -1)), axis=0)
            WeightArrays_, CoresInfo_ = nncompiler.apply(weight)
            
            self.WeightArrays.append(WeightArrays_)
            self.CoresInfo.append(CoresInfo_)
            
            InputPulse = inputcompiler.apply(Input)
            self.InputArray.append(InputPulse)

            maxCurrent = getrefv.apply(InputPulse, WeightArrays_, weight.shape[1])

            # maxCurrent = self.ReadVoltage * self.Gmax * self.numRow * 0.1 # Fix current for reduce compuitation

            self.MaxCurrent.append(maxCurrent)
            print("Start to calculate fc layer-%d" % (i+1))
            Input = layercal.DenseCal(InputPulse, WeightArrays_, maxCurrent, weight.shape[1])
            
            Input = activationfunc.apply(Input, "ReLU")

        self.Output = Input
        if (not(Label is None)):
            self.Accuracy = AccCal(Label, Input)
            
    def SaveMapData(self,
                    Weight,
                    Input,
                    WeightFile="MapWeight.csv",
                    InputFile="MapInput.csv"):
        """
        Save the inputs of input layer and weights after mapping
        """
        with open(WeightFile, "w") as f:
            writer = csv.writer(f)
            for i_layer in range(len(Weight)):
                for i_arrayV in range(len(Weight[i_layer])):
                    for i_arrayH in range(len(Weight[i_layer][i_arrayV])):
                        writer.writerow(("Layer", i_layer))
                        writer.writerow(("i_arrayV", i_arrayV))
                        writer.writerow(("i_arrayH", i_arrayH))
                        writer.writerow(("AD reference current", self.MaxCurrent[i_layer]))
                        writer.writerows((Weight[i_layer][i_arrayV][i_arrayH]))
#        with open(InputFile, "w") as f:
#            writer = csv.writer(f)
#            for i_batch in range(len(Input)):
#                for i_arrayV in range(len
    def HWEvaluate(self):
        '''
        Evaluate the hardware performance.
        '''
        # TODO add conv hweval in the future
        if self.numConv == 0:
            self.HWEval_.apply(self.BatchSize,
                    self.CoresInfo,
                    self.WeightArrays,
                    self.InputArray)

            # Traced date of the chip
            Area = self.HWEval_.Area
            ReadLatency = self.HWEval_.ReadLatency
            ReadPower = sum(self.HWEval_.ReadPower)/len(self.HWEval_.ReadPower)
            ReadDynamicEnergy = sum(self.HWEval_.ReadDynamicEnergy)/len(self.HWEval_.ReadDynamicEnergy)

            numCoreUsed = 0
            for i in range(len(self.CoresInfo)):
                numCoreUsed += self.CoresInfo[i][1] * self.CoresInfo[i][0]

            # A rough consideration for core schedule
            if self.numCoreHMax!=0 and self.numCoreVMax!=0:
                numCoreinChip = self.numCoreHMax * self.numCoreVMax
                if numCoreinChip > numCoreUsed:
                    self.numCore = numCoreUsed;
                    self.HWArea = numCoreinChip * Area
                    self.HWReadLatency = ReadLatency * max(len(self.CoresInfo), 
                            np.ceil(numCoreUsed/numCoreinChip))
                    self.HWReadPower = ReadPower * numCoreUsed
                    self.HWReadDynamicEnergy = ReadDynamicEnergy * numCoreUsed
                else:
                    self.numCore = numCoreinChip
                    self.HWArea = numCoreUsed * Area
                    self.HWReadLatency = len(self.CoresInfo)
                    self.HWReadPower = ReadPower * numCoreUsed
                    self.HWReadDynamicEnergy = ReadDynamicEnergy * numCoreUsed
            else:
                print("ERROR: [HWEval] Please specify the numCoreHMax and numCoreVMax!! ")
        
        numMVop = 0
        for i in range(self.numConv + self.numLinear):
            numMVop += self.CoresInfo[i][0] * self.CoresInfo[i][1]
        numOps = numMVop * self.numCol * self.numRow * 2 # 2 for add and multipy
        timeops = float(numOps)*1e-12/(self.HWReadLatency + self.ReadPulseWidth * self.IOBits*len(self.CoresInfo) * 1e-9)
        powerops = timeops/self.HWReadPower
        areaops = timeops/self.HWArea

        self.HWtimeops = timeops
        self.HWpowerops = powerops
        self.HWareaops = areaops


    def show(self):
        '''
        Print some data to console
        '''
        print("\n")
        print("---------------------------------")
        print("------------ Results ------------")
        print("---------------------------------")
        print("-Number of used Cores:", self.numCore)
        print("-RRAM Array Size:", self.numCol, "x", self.numRow)
        print("-Area:", self.HWArea*1e6, "mm^2")
        print("-ReadDynamicEnergy:", self.HWReadDynamicEnergy*1e9, "nJ/img")
        print("-ReadPower:", self.HWReadPower, "W")
        print("-Performace")
        print("|-Computing Performance:", self.HWtimeops, "TOPS")
        print("|-Energy Performance:", self.HWpowerops, "TOPS/W")
        print("|-Area Performance:", self.HWareaops, "TOPS/mm^2")
        print("-Accuracy:", self.Accuracy*100, "%")

if __name__ == "__main__":
    pass

