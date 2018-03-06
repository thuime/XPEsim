#include "HWsim.h"

int main()
{
	HWsim* hwsim;
	hwsim = new HWsim();
	hwsim->Initialize();
	hwsim->CalculateArea();
	hwsim->CalculateLatency();
	hwsim->CalculatePower();
	//hwsim->SaveOutput("HWoutput");
	hwsim->PrintProperty();

	return 0;
}
