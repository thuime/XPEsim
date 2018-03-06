// Sum.cpp
#include <cmath>
#include <iostream>
#include <stdlib.h>
#include "constant.h"
#include "formula.h"
#include "Sum.h"

using namespace std;

Sum::Sum(const InputParameter& _inputParameter,
	   	const Technology& _tech,
	   	const MemCell& _cell):
	FunctionUnit(),
   	inputParameter(_inputParameter),
   	tech(_tech),
   	cell(_cell),
   	adder(_inputParameter, _tech, _cell)
{
	initialized = false;
}

void Sum::Initialize(int _numCol, int _numCellPerSynapse, int _numSumBit)
{
	if (initialized)
		cout << "[Sum] Warning: Already initialized!" << endl;
	
	numCol = _numCol;
	numSumBit = _numSumBit;
	int numCellPerSynapse = _numCellPerSynapse;

	/*
	int numAdderPerCol = 0;
	if (numCellPerSynapse == 1 )
		numAdderPerCol = 0; // no need for Sum
	else
	{
		float n = float(numCellPerSynapse);
		while(n > 1)
		{
			numAdderPerCol += floor(n/2);
			n = ceil(n/2);
			numAdderDepth++;
		}//Calculate the number of adders in every column
	}
	*/
	if (numCellPerSynapse==0)
		numAdderDepth = 0;
	else
		numAdderDepth = log2(10) + 1;

	numAdder = numCol - numCellPerSynapse;// In every col, the output should substrate the bias
	adder.Initialize(numSumBit, numAdder);

	initialized = true;
}

void Sum::CalculateArea(double heightArray,  double widthArray, AreaModify _option) 
{
	if (!initialized) {
		cout << "[Sum] Error: Require initialization first!" << endl;
	} else {

		if (widthArray && _option == NONE)
		{
			adder.CalculateArea(NULL, widthArray, NONE);
		}else
		{
			cout << "[Sum] Error: No width assigned for the sum circuit!" << endl;
			exit(-1);
		}
		height = adder.height;
		width = adder.width;
		area = adder.area;

		// Modify layout
		newHeight = heightArray;
		newWidth = widthArray;
		switch (_option) {
			case MAGIC:
				MagicLayout();
				break;
			case OVERRIDE:
				OverrideLayout();
				break;
			default:    // NONE
				break;
		}
	}
}

void Sum::CalculateLatency(double numRead)
{
	if (!initialized) 
	{
		cout << "[Sum] Error: Require initialization first!" << endl;
	} else {
		readLatency = 0;
		adder.CalculateLatency(1e20, adder.capNandInput, 1);
		
		readLatency = adder.readLatency * numRead * numAdderDepth;
	}
}

void Sum::CalculatePower(double numRead) 
{
	if (!initialized) 
	{
		cout << "[Sum] Error: Require initialization first!" << endl;
	} else {
		leakage = 0;
		readDynamicEnergy = 0;
		

		adder.CalculatePower(numRead, numAdder);
		readDynamicEnergy += adder.readDynamicEnergy;
		leakage += adder.leakage;
		if (!readLatency)
			cout << "[Sum] Error: Need to calculate read latency first" << endl;
		else
			readPower = readDynamicEnergy/readLatency;
		
	}
}

void Sum::PrintProperty(const char* str) {
	FunctionUnit::PrintProperty(str);
}

void Sum::SaveOutput(const char* str, const char* outputFile)
{
    FunctionUnit::SaveOutput(str, outputFile);
}
