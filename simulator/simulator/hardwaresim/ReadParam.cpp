// ReadParam.cpp
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "ReadParam.h"

using namespace std;

ReadParam::ReadParam() 
{
	/* Default parameter */
	designopt = ReadLatency_Opt;
	designoptname = "Read Latency";
	outputFilePrefix = "HWoutput";
	clkFreq = 2e9;
	wireWidth = 40;
	fileCellType = Type::DigitalRRAMTHU;

	processNode = 65;
	deviceRoadmap = HP;
	transistorType = conventional;
	temperature = 301;
	
	Rmin = 25e3;
	Rmax = 265e3;
	CellBits = 4;
	WeightBits = 8;

	ResetVoltage = 1.7;
	SetVoltage = 1.7;
	ReadVoltage = 0.15;
	ResetPulseWidth = 50e-9;
	SetPulseWidth = 50e-9;
	ReadPulseWidth = 50e-9;
	accessVoltage = 1.3;

	multipleCells = 1;

	numArrayCol = 64;
	numArrayRow = 64;
	IOBits = 8;
	SABits = 4;
}

void ReadParam::ReadParameterFromFile(const string & inputFile)
{
	FILE *fp = fopen(inputFile.c_str(), "r");
	char line[5000];
	char tmp[5000];

	if (!fp)
	{
		cout << inputFile << " cannot be found!" << endl;
		cout << "Using the default parameter!" << endl;
	}
	else
	{

		while (fscanf(fp, "%[^\n]\n", line) != EOF)
		{
	
			if (!strncmp("-DesignOptimization", line, strlen("-OptimizationTarget")))
			{
				sscanf(line, "-OptimizationTarget %s", tmp);
				if (!strcmp(tmp, "ReadLatency"))
				{
					designopt = ReadLatency_Opt;
					designoptname = "ReadLatency_Opt";
				}
				else if (!strcmp(tmp, "WriteLatency"))
					designopt = WriteLatency_Opt;
				else if (!strcmp(tmp, "ReadDynamicEnergy"))
					designopt = ReadDynamicEnergy_Opt;
				else if (!strcmp(tmp, "WriteDynamicEnergy"))
					designopt = WriteDynamicEnergy_Opt;
				else if (!strcmp(tmp, "Leakage"))
					designopt = Leakage_Opt;
				else if (!strcmp(tmp, "Area"))
					designopt = Area_Opt;
				else if (!strcmp(tmp, "None"))
					designopt = None;
				else
					designopt = FULL_Opt;
				continue;
			}
	
			if (!strncmp("-OutputFilePrefix", line, strlen("-OutputFilePrefix"))) 
			{
				sscanf(line, "-OutputFilePrefix %s", tmp);
				outputFilePrefix = (string)tmp;
				continue;
			}
	
			if (!strncmp("-ProcessNode", line, strlen("-ProcessNode")))
			{
				sscanf(line, "-ProcessNode %d", &processNode);
				continue;
			}
	
			if (!strncmp("-CellType", line, strlen("-CellType")))
			{
				sscanf(line, "-CellType %s", tmp);
				fileCellType = (string)tmp;
				continue;
			}
	
			if (!strncmp("-WeightBits", line, strlen("-WeightBits")))
			{
				sscanf(line, "-WeightBits %d", &WeightBits);
			   continue;
			}
	
			if (!strncmp("-CellBits", line, strlen("-CellBits")))
			{
				sscanf(line, "-CellBits %d", &CellBits);
			   continue;
			}
	
			if (!strncmp("-ReadVoltage", line, strlen("-ReadVoltage")))
			{
				sscanf(line, "-ReadVoltage %lf", &ReadVoltage);
			   continue;
			}
	
			if (!strncmp("-numArrayCol", line, strlen("numArrayCol")))
			{
				sscanf(line, "-numArrayCol %d", &numArrayCol);
			   continue;
			}
	
			if (!strncmp("-numArrayRow", line, strlen("-numArrayRow")))
			{
				sscanf(line, "-numArrayRow %d", &numArrayRow);
			   continue;
			}
	
			if (!strncmp("-IOBits", line, strlen("-IOBits")))
			{
				sscanf(line, "-IOBits %d", &IOBits);
			   continue;
			}
		}
		fclose(fp);
	}

}

void ReadParam::PrintParameter() 
{

	cout << "---------------------------------------------------------" << endl
		<< "--------------- A Summary of Parameters -----------------" << endl
		<< "---------------------------------------------------------" << endl
	
		<< "--Design Optimation: " << designoptname << endl
		<< "--OutputFile: " << outputFilePrefix << endl
		<< "--Clock Frequence: " << clkFreq/1e9 << " GHz" << endl
		<< "--CellType: " << fileCellType << endl
		<< endl
		<< "--Process Node: " << processNode << " nm" <<  endl
		//<< "--deviceRoadmap: " << deviceRoadmap << endl
		//<< "--transistorType: " << transistorType << endl
		<< "--Temperature: " << temperature << " K" << endl
		<< endl
		<< "--Rmin: " << Rmin/1e3 << " Kohm" << endl
		<< "--Rmax: " << Rmax/1e3 << " Kohm" << endl
		<< "--Cell Bits: " << CellBits << endl
		<< "--Weight Bits: " << WeightBits << endl
		<< endl
		<< "--Pulse Setup:" << endl
		<< " |--Reset Voltage: " << ResetVoltage << " V" << endl
		<< " |--Set Voltage: " << SetVoltage << " V" << endl
		<< " |--Read Voltage: " << ReadVoltage << " V" << endl
		<< " |--Reset Pulse Width: " << ResetPulseWidth/1e-9 << " ns" << endl
		<< " |--Set Pulse Width: " << SetPulseWidth/1e-9 << " ns" << endl
		<< " |--Read Pulse Width: " << ReadPulseWidth/1e-9 << " ns" << endl
		<< " |--Access Voltage: " << accessVoltage << " V" << endl
		<< endl
		//<< "--multipleCells"
		<< "--Array Column Size: " << numArrayCol << endl
		<< "--Array Row Size: " << numArrayRow << endl
		<< "--IO Bits: " << IOBits << endl
		<< "--AD Bits: " << SABits << endl
		<< "--------------------- End of Summary --------------------" << endl;
}
