from distutils.core import setup, Extension

import os 
os.system("swig3.0 -c++ -python HWsim.i")

pht_module = Extension("_HWsim",
        sources=["HWsim_wrap.cxx",
            "ReadParam.cpp",
            "Technology.cpp",
            "Sum.cpp",
            "FunctionUnit.cpp",
            "formula.cpp",
            "Adder.cpp",
            "RowDecoder.cpp",
            "Mux.cpp",
            "WLDecoderOutput.cpp",
            "DFF.cpp",
            "DeMux.cpp",
            "VoltageSenseAmp.cpp",
            "Precharger.cpp",
            "SenseAmp.cpp",
            "DecoderDriver.cpp",
            "SRAMWriteDriver.cpp",
            "ReadCircuit.cpp",
            "ShiftAdd.cpp",
            "SwitchMatrix.cpp",
            "WLNewDecoderDriver.cpp",
            "NewSwitchMatrix.cpp",
            "CurrentSenseAmp.cpp",
            "Comparator.cpp",
            "MultilevelSAEncoder.cpp",
            "MultilevelSenseAmp.cpp",
            "SubArray.cpp",
            "Core.cpp",
            "HWsim.cpp"
            ],
        )
setup(name = "HWsim",
        version = "0.1",
        author = "THU.IME",
        description = "HWsim",
        ext_modules = [pht_module],
        py_modules = ["HWsim"],
        )

