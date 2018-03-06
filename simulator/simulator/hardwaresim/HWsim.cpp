// HWsim.cpp
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include "HWsim.h"

using namespace std;

HWsim::HWsim()
{
	initialized = false;
}

void HWsim::Initialize()
{
	if (initialized)
		cout << "[HWsim] Warning: Already initialized! " << endl;

	/* Read parameter from file */
	params.ReadParameterFromFile("simconfig");
	
	/* cell configuration */
	cell.memCellType = Type::RRAM;
	cell.accessType = CMOS_access;
	cell.resistanceOn = params.Rmax;
	cell.resistanceOff = params.Rmin;
	cell.resistanceAvg = (cell.resistanceOn + cell.resistanceOff)/2;
	cell.readVoltage = params.ReadVoltage;
	double writeVoltageLTP = params.ResetVoltage;
	double writeVoltageLTD = params.SetVoltage;
	cell.writeVoltage = sqrt(writeVoltageLTP * writeVoltageLTP + writeVoltageLTD * writeVoltageLTD);
	cell.readPulseWidth = params.ReadPulseWidth;
	double writePulseWidthLTP = params.ResetPulseWidth;
	double writePulseWidthLTD = params.SetPulseWidth;	
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
	cell.accessVoltage = params.accessVoltage;
	cell.multipleCells = params.multipleCells;
	cell.heightInFeatureSize = (cell.accessType==CMOS_access)? 4 : 2;
	cell.widthInFeatureSize = (cell.accessType==CMOS_access)? 4 : 2;
	cell.featureSize = 40e-9;

	/* input parameter and technology configuration */
	inputParameter.transistorType = params.transistorType;
	inputParameter.deviceRoadmap = params.deviceRoadmap;
	inputParameter.temperature = params.temperature;
	inputParameter.processNode = params.processNode;
	tech.Initialize(inputParameter.processNode, inputParameter.deviceRoadmap, inputParameter.transistorType);
	
	/*  Initialize interconnect wires for IR drop*/
	double AR;	/* Aspect ratio of wire height to wire width */
	double Rho;	/* Resistivity */
	double wireResistanceCol, wireResistanceRow, unitLengthWireResistance;
	wireWidth = params.wireWidth;
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
	double wireLength = wireWidth * 1e-9 * 2;	/* 2F */
	if (wireWidth == -1)
	{
		unitLengthWireResistance = 1.0;	/* Use a small number to prevent numerical error for NeuroSim */
		wireResistanceRow = 0;
		wireResistanceCol = 0;
	}
	else
	{
		unitLengthWireResistance =  Rho / ( wireWidth*1e-9 * wireWidth*1e-9 * AR );
		wireResistanceRow = unitLengthWireResistance * wireLength;
		wireResistanceCol = unitLengthWireResistance * wireLength;
	}

	/* Core initialization */
	core = new Core(inputParameter, tech, cell, params);
	core->Initialize(unitLengthWireResistance);

	initialized = true;
}

void HWsim::CalculateArea()
{
	if (!initialized)
		cout << "[HWsim] Error: Require initialization first!" << endl; // To ensure initialization first
	else
	{
		core->CalculateArea();
		area = core->area;
	}
}

void HWsim::CalculateLatency()
{
	if (!initialized)
		cout << "[HWsim] Error: Require initialization first!" << endl; // To ensure initialization first
	else
	{
		core->CalculateLatency(1e20);
		readLatency = core->readLatency;
	}
}

void HWsim::CalculatePower(double activityRowRead)
{
	if (!initialized)
		cout << "[HWsim] Error: Require initialization first!" << endl; // To ensure initialization first
	else
	{
		readPower = 0;
		readDynamicEnergy = 0;
		writeDynamicEnergy = 0;

		core->numof1 = numof1;core->numof2 = numof2;core->numof3 = numof3;
		core->numof4 = numof4;core->numof5 = numof5;core->numof6 = numof6;
		core->numof7 = numof7;core->numof8 = numof8;core->numof9 = numof9;
		core->numof10 = numof10;
		core->CalculatePower(activityRowRead);
		readPower = core->readPower;
		readDynamicEnergy = core->readDynamicEnergy;
	}
}

void HWsim::SaveOutput(int CoreIndex, const char* outputFile)
{
	ofstream outfile;
	outfile.open(outputFile, ios::app);
	outfile << "---------------------------------------------------------" << endl;
	outfile << "---------- Neural Network Processor Unit Design ---------" << endl; 
	outfile << "---------------------------------------------------------" << endl;
	core->SaveOutput(CoreIndex, outputFile);
	outfile << endl;
	outfile.close();
}


void HWsim::PrintProperty()
{
	cout << "---------------------------------------------------------" << endl
		<< "---------- Neural Network Processor Unit Design ---------" << endl
		<< "---------------------------------------------------------" << endl;
	params.PrintParameter();
	cout << "---------------------------------------------------------" << endl
		<< "--------------------- Final Design ----------------------" << endl
		<< "---------------------------------------------------------" << endl;

	core->PrintProperty();
}

