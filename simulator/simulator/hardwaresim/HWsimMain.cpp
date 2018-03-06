#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include "InputParameter.h"
#include "Technology.h"
#include "MemCell.h"
#include "ReadParam.h"
#include "Core.h"

using namespace std;

int main()
{
	Core *core;
	InputParameter inputParameter;
	Technology tech;
	MemCell cell;
	ReadParam readParam;
	int wireWidth = 40;
	/* Read parameter from file */
	readParam.ReadParameterFromFile("simconfig");
	
	/* cell configuration */
	cell.memCellType = Type::RRAM;
	cell.accessType = CMOS_access;
	cell.resistanceOn = 25e3;
	cell.resistanceOff = 265e3;
	cell.resistanceAvg = (cell.resistanceOn + cell.resistanceOff)/2;
	cell.readVoltage = 0.15;
	double writeVoltageLTP = 2;
	double writeVoltageLTD = 2;
	cell.writeVoltage = sqrt(writeVoltageLTP * writeVoltageLTP + writeVoltageLTD * writeVoltageLTD);
	cell.readPulseWidth = 10e-9;
	double writePulseWidthLTP = 50e-9;
	double writePulseWidthLTD = 50e-9;	
	cell.writePulseWidth = (writePulseWidthLTP + writePulseWidthLTD) / 2;
	cell.nonlinearIV = false;
	cell.nonlinearity = 10;
	if (cell.nonlinearIV)
   	{
		double Vr_exp = 1;
		cell.resistanceOn = NonlinearResistance(cell.resistanceOn, cell.nonlinearity, cell.writeVoltage, Vr_exp, cell.readVoltage);
		cell.resistanceOff = NonlinearResistance(cell.resistanceOff, cell.nonlinearity,cell.writeVoltage, Vr_exp, cell.readVoltage);
		cell.resistanceAvg = (cell.resistanceOn + cell.resistanceOff)/2;
	}
	cell.accessVoltage = 1.4;
	int multipleCells = 1;
	cell.multipleCells = multipleCells;
	cell.heightInFeatureSize = (cell.accessType==CMOS_access)? 4 : 2;
	cell.widthInFeatureSize = (cell.accessType==CMOS_access)? 4 : 2;
	cell.featureSize = 40e-9;

	/* input parameter and technology configuration */
	inputParameter.transistorType = conventional;
	inputParameter.deviceRoadmap = HP;
	inputParameter.temperature = 301;
	inputParameter.processNode = 130;
	tech.Initialize(inputParameter.processNode, inputParameter.deviceRoadmap, inputParameter.transistorType);

	double AR;
	double Rho;
	double wireResistanceCol, wireResistanceRow, unitLengthWireResistance;
	switch(wireWidth)
   	{
		case 200:   AR = 2.10; Rho = 2.42e-8; break;
		case 100:   AR = 2.30; Rho = 2.73e-8; break;
		case 50:    AR = 2.34; Rho = 3.91e-8; break;
		case 40:    AR = 1.90; Rho = 4.03e-8; break;
		case 32:    AR = 1.90; Rho = 4.51e-8; break;
		case 22:    AR = 2.00; Rho = 5.41e-8; break;
		case 14:    AR = 2.10; Rho = 7.43e-8; break;
		case -1:    break;  // Ignore wire resistance or user define
		default:    exit(-1); puts("Wire width out of range");
	}
	double wireLength = wireWidth * 1e-9 * 2;
	if (wireWidth == -1)
	{
		unitLengthWireResistance = 1.0;
		wireResistanceRow = 0;
		wireResistanceCol = 0;
	}
	else
	{
		unitLengthWireResistance =  Rho / ( wireWidth*1e-9 * wireWidth*1e-9 * AR );
		wireResistanceRow = unitLengthWireResistance * wireLength;
		wireResistanceCol = unitLengthWireResistance * wireLength;
	}

	core = new Core(inputParameter, tech, cell, readParam);
	core->Initialize(unitLengthWireResistance);


	core->CalculateArea();
	core->CalculateLatency(1e20);
	core->CalculatePower();
	core->PrintProperty();
	
	//core->SaveOutput(readParam.outputFilePrefix);

	return 0;
}
