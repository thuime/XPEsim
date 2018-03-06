#sysfunctional.py
from __future__ import print_function, division
import numpy as np

def AccCal(Label, Output):
        Result = []
        for sample in Output:
            MaxOutput = np.argmax(sample)
            Result.append(MaxOutput)
        Result = np.asarray(Result)
        AccTmp = Result - Label
        Accuracy = 1.0 - np.nonzero(AccTmp)[0].shape[0]/AccTmp.shape[0]
        return Accuracy


