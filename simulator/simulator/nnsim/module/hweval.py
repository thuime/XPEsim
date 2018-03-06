# hwenval.py

from __future__ import print_function, division
from ...hardwaresim.HWsim import HWsim
import numpy as np

class HWEval(object):

    def __init__(self):
        super(HWEval, self).__init__()
        self.HWsim_ = HWsim()
        self.HWsim_.Initialize()
        self.ReadPower = []
        self.ReadDynamicEnergy = []
        self.Area = 0
        self.ReadLatency = 0

    def apply(self,
            BatchSize,
            CoresInfo,
            WeightCompiled,
            InputCompiled):

        """
        Evaluate the hardware performance.
        require args:
            BatchSize: the size of a batch
            CoresInfo: the information of splited cores
            WeightCompiled: the compiled weights(numLayrs*numCoreV*numCoreH*numRow*numCol)
            InputCompiled: the compiled Inputpulse (numLayers*BatchSize*numCoreV*numBits*numRow*1)
        """

        self.HWsim_.CalculateArea()
        self.HWsim_.CalculateLatency()
        numLayers = len(CoresInfo)
        numBits = len(InputCompiled[0][0][0])
        numRow = len(InputCompiled[0][0][0][0])
        # possible column       effective resistance under condition : 25k/100M, 64*64 array
        Rcolmin = 390.
        Rcolmax = 100e6
        Rlevel = 10 #  10 circuit simulation situation
        Rstep = (Rcolmax-Rcolmin) / (Rlevel-1)

        ReadPower = []
        for i in range(numLayers):
            numCoreV, numCoreH = CoresInfo[i]
            for j in range(BatchSize):
                print("numLayer:", i, "Input pic:", j)
                for m in range(numCoreV):
                    for bit in range(numBits):
                        Inputpulse = InputCompiled[i][j][m][bit]
                        activeread = (sum(InputCompiled[i][j][m][bit])/numRow)[0]
                        for n in range(numCoreH):
                            ArrayWeight = WeightCompiled[i][m][n]
                            ResCol = 1 / (np.dot(Inputpulse.T, ArrayWeight)+1e-20)
                            LevelCol = np.floor((ResCol-Rcolmin)/Rstep + 0.5) + 1
                            LevelCol = LevelCol.astype('int')
                            numoflevels = []
                            for level in range(Rlevel):
                                numoflevels.append(np.sum(LevelCol==level))
                            self.HWsim_.numof1 = numoflevels[0]
                            self.HWsim_.numof2 = numoflevels[1]
                            self.HWsim_.numof3 = numoflevels[2]
                            self.HWsim_.numof4 = numoflevels[3]
                            self.HWsim_.numof5 = numoflevels[4]
                            self.HWsim_.numof6 = numoflevels[5]
                            self.HWsim_.numof7 = numoflevels[6]
                            self.HWsim_.numof8 = numoflevels[7]
                            self.HWsim_.numof9 = numoflevels[8]
                            self.HWsim_.numof10 = numoflevels[9]
                            self.HWsim_.CalculatePower(activeread)
                            self.ReadPower.append(self.HWsim_.readPower)
                            self.ReadDynamicEnergy.append(self.HWsim_.readDynamicEnergy)
                            self.Area = self.HWsim_.area
                            self.ReadLatency = self.HWsim_.readLatency


                            #self.HWsim_.SaveOutput(n+m*numCoreV,"HWoutput")
