python setup.py build
cp build/lib.linux-x86_64-2.7/HWsim.py .
cp build/lib.linux-x86_64-2.7/_HWsim.so .
python ./testHW.py
