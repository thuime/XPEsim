# ParameterInput.py
from __future__ import print_function, division
import os
import numpy as np

class Parameterinput(object):
    '''
    Read the input parameters from simconfig file
    '''

    def __init__(self):
        # Default parameter
        self.Gmax = 1/25e3
        self.Gmin = 1/265e3
        self.numCol = 128
        self.numRow = 128
        self.ReadVoltage = 0.15 # unit: V
        self.WeightBits = 8
        self.CellBits = 4
        self.IOBits = 8
        self.ADBits = 8
        
        
        # Read parameter from file
        if not os.path.isfile("simconfig"):
            raise IOError("[SimConfig] There is no parameter config file.")
        with open("simconfig", "r") as fi:
            f = fi.read()
            a = f.find("CellType", 0)
            b = f.find("\n", a)
            CellType  = f[a+len("CellType")+1:b]
            if CellType == "DigitalRRAMTHU":
                self.params = {}
                paramlist = ["CellType", "WeightBits", "CellBits",
                        "Rmax", "Rmin", "ReadVoltage", "IOBits",
                        "numArrayCol", "numArrayRow", "numCoreVMax",
                        "numCoreHMax"]
                for i in paramlist:
                    a = f.find("-"+i, 0)
                    b = f.find("\n", a)
                    paramvalue = f[a+len(i)+1:b]
                    self.params[i] = paramvalue
        self.Gmax = 1/float(self.params["Rmin"])
        self.Gmin = 1/float(self.params["Rmax"])
        self.ReadVoltage = float(self.params["ReadVoltage"])
        self.numCol = int(self.params["numArrayCol"])
        self.numRow = int(self.params["numArrayRow"])
        self.WeightBits = int(self.params["WeightBits"])
        self.CellBits = int(self.params["CellBits"])
        self.IOBits = int(self.params["IOBits"])
        self.numCoreVMax = int(self.params["numCoreVMax"]) # Define the max num of veritcal core used on chip
        self.numCoreHMax = int(self.params["numCoreHMax"]) # Define the max num of horizontal core used on chip
        self.ReadPulseWidth = 50 #unit: ns
        self.numCellperWeight = int(np.ceil(self.WeightBits/self.CellBits))
        self.numLayerOutput = 0 # The number of outputs in the layer
        self.RangeMax = 2 ** (self.WeightBits - 1) # The range of bitwise weight is from -self.RangeMax to (self.RangeMax-1)
