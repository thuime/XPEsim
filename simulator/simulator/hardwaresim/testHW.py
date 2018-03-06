import shutil, HWsim

print "11"
a = HWsim.HWsim()
print "22"
a.Initialize()
print "3"
for i in [0.1, 0.3, 0.5]:
    a.CalculateArea()
    print "4"
    a.CalculateLatency()
    print "5"
    a.CalculatePower(i)
    print "6"

 #   a.PrintProperty()

    a.SaveOutput(1, "HWoutput")

#print a.area

#a.PrintProperty()
