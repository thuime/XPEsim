// Core.cpp
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include "Core.h"

using namespace std;

Core::Core(InputParameter& _inputParameter, Technology& _tech,
	   	MemCell& _cell, ReadParam& _readParam):
	inputParameter(_inputParameter),
   	tech(_tech),
   	cell(_cell),
	readParam(_readParam),
	sum_add(_inputParameter, _tech, _cell),
	shiftAdd(_inputParameter, _tech, _cell),
	subArray(_inputParameter, _tech, _cell)
{
	initialized = false;
}

void Core::Initialize(double _unitWireRes)
{
	if (initialized)
		cout << "[Core] Warning: Already initialized! " << endl;

	numArrayCol = readParam.numArrayCol;
	numArrayRow = readParam.numArrayRow;
	numWeightBit = readParam.WeightBits;
	numCellBit = readParam.CellBits;
	numCellPerSynapse = numWeightBit*1.0 / numCellBit;
	numBitInput = readParam.IOBits;
	numOutput = floor(numArrayCol/numCellPerSynapse)-1; // The output number, -1 means substrate bias
	numSABit = readParam.SABits;
	/* Initialize Sum */
	sum_add.Initialize(numArrayCol, numCellPerSynapse, numSABit);

	/* Initialize ShiftAdder */
	int ShiftAddmux = 10.0;
	shiftAdd.Initialize(ceil(numOutput/ShiftAddmux), readParam.IOBits, readParam.clkFreq, NONSPIKING, numBitInput);

	/* Initialize SubArray */
	subArray.activityRowWrite = (double)1/2; 
	subArray.activityColWrite = (double)1/2;

	subArray.spikingMode = NONSPIKING;
	if (subArray.spikingMode == SPIKING) 
		subArray.numReadPulse = pow(2, numBitInput);
	else
		subArray.numReadPulse = numBitInput;

	subArray.numWritePulse = 1;
	subArray.neuro = 1;
	subArray.multifunctional = 0;
	subArray.digitalModeNeuro = 0;
	subArray.newBNNrowbyrowMode = 0;
	subArray.newBNNparallelMode = 0;


	subArray.tsqinghua = 1;
	if (1 <= numSABit && numSABit<= 6)
		subArray.levelOutput = pow(2, numSABit);
	else
	{
		subArray.levelOutput = pow(2, 6);
		cout << "[Core] Error: Current version only support SABit=1-6!" << endl;
		}

	subArray.FirstLayer = 0;
	subArray.XNORModeDoubleEnded = 0;
	subArray.XNORModeSingleEnded = 0;
	subArray.clkFreq = readParam.clkFreq;
	subArray.parallelWrite = false;
	subArray.FPGA = false;
	subArray.LUT_dynamic = false;
	subArray.backToBack = false;
	subArray.numLut = 32;
	subArray.numColMuxed = 4;
	subArray.numWriteColMuxed = 4;

	if (subArray.spikingMode == NONSPIKING && subArray.numReadPulse > 1)
		subArray.shiftAddEnable = true;
	else
		subArray.shiftAddEnable = false;
	subArray.relaxArrayCellHeight = 0;
	subArray.relaxArrayCellWidth = 0;

	subArray.readCircuitMode = CMOS;
	subArray.maxNumIntBit = numBitPartialSum;

	int numBitLTP = numWeightBit;
	int numBitLTD = numWeightBit;
	int maxNumLevelLTP = pow(2, numBitLTP) - 1;
	int maxNumLevelLTD = pow(2, numBitLTD) - 1;

	subArray.maxNumWritePulse = (maxNumLevelLTP > maxNumLevelLTD)? maxNumLevelLTP : maxNumLevelLTD;

	if (subArray.digitalModeNeuro)
	{
		subArray.numCellPerSynapse = numWeightBit;
		subArray.avgWeightBit = subArray.numCellPerSynapse;
	}
	else if(subArray.newBNNparallelMode)
	{
		subArray.numCellPerSynapse = numWeightBit;
		subArray.avgWeightBit = subArray.numCellPerSynapse;
	}
	else if(subArray.newBNNrowbyrowMode)
	{
		subArray.numCellPerSynapse = 1;
		subArray.avgWeightBit = subArray.numCellPerSynapse;
	}
	else if(subArray.XNORModeDoubleEnded)
	{
		subArray.numCellPerSynapse = numWeightBit;
		subArray.avgWeightBit = subArray.numCellPerSynapse;
	}
	else if(subArray.XNORModeSingleEnded)
	{
		subArray.numCellPerSynapse = 1;
		subArray.avgWeightBit = subArray.numCellPerSynapse;
	}
	else if(subArray.tsqinghua) // Tsinghua Design
	{
		subArray.numCellPerSynapse = numCellPerSynapse;
		subArray.avgWeightBit = subArray.numCellPerSynapse;
	}
	else
	{
		subArray.numCellPerSynapse = 1;
		subArray.avgWeightBit = (numBitLTP > numBitLTD)? numBitLTP : numBitLTD;
	}

	int numRow;
	if (subArray.XNORModeDoubleEnded && !subArray.FPGA)
		numRow = numArrayRow * 2;
	else if (subArray.XNORModeSingleEnded && !subArray.FPGA)
		numRow = numArrayRow * 2;
	else
		numRow = numArrayRow;

	int numCol;
	if (subArray.neuro && !subArray.FPGA)
	{
		numCol = numArrayCol * subArray.numCellPerSynapse;
	}
	else
		numCol = numArrayCol;


	if (subArray.numColMuxed > numCol)
		subArray.numColMuxed = numCol;

	subArray.numReadCellPerOperationFPGA = numCol;
	subArray.numWriteCellPerOperationFPGA = numCol;
	subArray.numReadCellPerOperationMemory = numCol;

	subArray.numWriteCellPerOperationMemory = numCol/8;
	subArray.numReadCellPerOperationNeuro = numCol * subArray.numCellPerSynapse;
	subArray.numWriteCellPerOperationNeuro = numCol;

	//subArray.activityRowRead = activityRowRead; //(double) numofreadrow/numRow;

	subArray.numof1=numof1, subArray.numof2=numof2, subArray.numof3=numof3, subArray.numof4=numof4,
		subArray.numof5=numof5, subArray.numof6=numof6, subArray.numof7=numof7,
		subArray.numof8=numof8,subArray.numof9=numof9, subArray.numof10=numof10;

	subArray.Initialize(numArrayRow, numArrayCol, _unitWireRes);


	initialized = true;
}

void Core::CalculateArea()
{
	if (!initialized)
		cout << "[Core] Error: Require initialization first!" << endl; // To ensure initialization first
	else
	{
		subArray.CalculateArea();
		width = subArray.width;
		sum_add.CalculateArea(NULL, width, NONE);
		shiftAdd.CalculateArea(NULL, width, NONE);
		height = subArray.height + sum_add.height + shiftAdd.height;

		area = height * width;
		usedArea = subArray.usedArea + sum_add.area + shiftAdd.area;
		emptyArea = area - usedArea;
	}
}

void Core::CalculateLatency(double _rampInput)
{
	if (!initialized)
		cout << "[Core] Error: Require initialization first!" << endl; // To ensure initialization first
	else
	{
		subArray.CalculateLatency(_rampInput);
		sum_add.CalculateLatency(numBitInput);
		shiftAdd.CalculateLatency(numBitInput);

		readLatency = 0;
		readLatency += subArray.readLatency;
		readLatency += sum_add.readLatency;
		readLatency += shiftAdd.readLatency;

		writeLatency = 0;
		writeLatency = subArray.writeLatency;
		
	}
}

void Core::CalculatePower(double activityRowRead)
{
	if (!initialized)
		cout << "[Core] Error: Require initialization first!" << endl; // To ensure initialization first
	else
	{
		subArray.numof1 = numof1;
		subArray.numof2 = numof2;
		subArray.numof3 = numof3;
		subArray.numof4 = numof4;
		subArray.numof5 = numof5;
		subArray.numof6 = numof6;
		subArray.numof7 = numof7;
		subArray.numof8 = numof8;
		subArray.numof9 = numof9;
		subArray.numof10 = numof10;
		subArray.CalculatePower(activityRowRead);
		sum_add.CalculatePower(numBitInput);
		shiftAdd.CalculatePower(numBitInput);

		readDynamicEnergy = 0;
		readDynamicEnergy += subArray.readDynamicEnergy;
		readDynamicEnergy += sum_add.readDynamicEnergy;
		readDynamicEnergy += shiftAdd.readDynamicEnergy;

		writeDynamicEnergy = 0;
		writeDynamicEnergy += subArray.writeDynamicEnergy;

		leakage = 0;
		leakage += subArray.leakage;
		leakage += sum_add.leakage;
		leakage += shiftAdd.leakage;

		if (!readLatency)
			cout << "[Core] Error: Need to calculate read latency first" << endl;
		else
			readPower = readDynamicEnergy/readLatency + leakage;

		if (!writeLatency)
			cout << "[Core] Error: Need to calculate write latency first" << endl;
		else 
			writePower = writeDynamicEnergy/writeLatency + leakage;

	}
}

void Core::SaveOutput(const int& CoreIndex, const char* outputFile)
{
	ofstream outfile;
	outfile.open(outputFile, ios::app);
	outfile << " CoreIndex: " << CoreIndex << endl;
	subArray.SaveOutput(outputFile);
	sum_add.SaveOutput("Sum", outputFile);
	shiftAdd.SaveOutput("ShiftAdd", outputFile);
	FunctionUnit::SaveOutput("Calculation Core", outputFile);
	outfile << "Used Area = " << usedArea*1e12 << "um^2" << endl;
	outfile << "Empty Area = " << emptyArea*1e12 << "um^2" << endl;
	outfile << '\n' << endl;
	outfile << endl;
	outfile.close();
}


void Core::PrintProperty()
{
	subArray.PrintProperty();
	sum_add.PrintProperty("Sum");
	shiftAdd.PrintProperty("ShiftAdd");
	FunctionUnit::PrintProperty("Core");
	cout << "Used Area = " << usedArea*1e12 << "um^2" << endl
		<< "Empty Area = " << emptyArea*1e12 << "um^2" << endl
		<< '\n' << endl;
}

