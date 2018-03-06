// ActivationFunc.cpp
#include <cmath>
#include <iostream>
#include <stdlib.h>
#include "constant.h"
#include "formula.h"
#include "ActivationFunc.h"

using namespace std;

ActivationFunc::ActivationFunc(const InputParameter& _inputParameter,
	   	const Technology& _tech,
	   	const MemCell& _cell,
	   	const ReadParam& _readParam):
   	FunctionUnit(),
   	inputParameter(_inputParameter),
   	tech(_tech),
   	cell(_cell),
   	readParam(_readParam)
{
	initialized = false;
}

void ActivationFunc::Initialize(int _numCol, int _numIOBit, ActivationFunction _Func)
{
	if (initialized)
		cout << "[ActivationFunc] Warning: Already initialized!" << endl;
	
	numCol = _numCol;
	numIOBit = _numIOBit;
	Func = _Func;

	widthNandN = 2 * MIN_NMOS_SIZE * tech.featureSize;
	widthNandP = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;
	widthInvN = MIN_NMOS_SIZE * tech.featureSize;
	widthInvP = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;

	initialized = true;
}

void ActivationFunc::CalculateArea(double numInput, double IOBits,
	   	double _newHeight, double _newWidth,
	   	AreaModify _option) 
{
	if (!initialized) 
	{
		cout << "[ActivationFunc] Error: Require initialization first!" << endl;
		exit(-1);
	}
	
	switch (Func)
	{
		case ReLU:
			{
				// assume ReLU consists of 2 Inv and 1 Nand
				CalculateGateArea(NAND, 2, widthNandN, widthNandP, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hNand, &wNand);
				CalculateGateArea(INV, 1, widthInvN, widthInvP, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hInv, &wInv);

				int numInv = 2 * IOBits * numInput;
				int numNand = IOBits * numInput;

				int numInvPerRow, numNandPerRow;
				int numInvPerCol, numNandPerCol;
				if (_newWidth && _option == NONE)
				{
					numInvPerRow = (int) floor(_newWidth/wInv);
					numNandPerRow = (int) floor(width/wNand);
					numInvPerCol = (int) ceil((float)numInv/numInvPerRow);
					numNandPerCol = (int) ceil((float)numNand/numNandPerRow);

					height = numInvPerCol * hInv + numNandPerCol * hNand;
					width = max( numInvPerRow * wInv, numNandPerRow * wNand);

					area = height * width;
				}
				else
				{
					cout << "[ActivationFunc] Error: Require specify _newWidth!"
						<< endl;
				}
			}

		case Sigmoid:
			{
				cout << "[ActivationFunc] Error: Sigmoid is still under development!" 
					<< endl;
			}

		case Tanh:
			{
				cout << "[ActivationFunc] Error: Tanh is still under development!"
					<< endl;
			}

		default:
			{
				cout << "[ActivationFunc] Error: Please specify the activation function"
					<< endl;
			}
	}

		// Modify layout
	newHeight = _newHeight;
	newWidth = _newWidth;
	switch (_option) 
	{
		case MAGIC:
			MagicLayout();
			break;
		case OVERRIDE:
			OverrideLayout();
			break;
		default:    // NONE
			break;
	}

	// Calculate INV/NAND2 capacitance
	CalculateGateCapacitance(INV, 1, widthInvN, widthInvP, hInv, tech, &capInvInput,&capInvOutput); 
	CalculateGateCapacitance(NAND, 2, widthNandN, widthNandP, hNand, tech, &capNandInput, &capNandOutput);

}

void ActivationFunc::CalculateLatency(double _rampInput, double _capLoad, double numRead)
{
	if (!initialized) 
	{
		cout << "[ActivationFunc] Error: Require initialization first!" << endl;
		exit(-1);
	}

	switch (Func)
	{
		case ReLU:
		{
			capLoad = _capLoad;
			rampInput = _rampInput;
			readLatency = 0;
	
			double resPullDown;
			double tr;      /* time constant */
			double gm;      /* transconductance */
			double beta;    /* for horowitz calculation */
			double rampInvOutput = 1e20;
			double rampNandOutput = 1e20;
			
			//INV
			resPullDown = CalculateOnResistance(widthInvN, NMOS, inputParameter.temperature, tech); // Why do not consider Up/Down....
			tr = resPullDown * (capInvOutput + capNandInput);
			gm = CalculateTransconductance(widthInvN, NMOS, tech);
			beta = 1 / (resPullDown * gm);
			readLatency += horowitz(tr, beta, rampInput, &rampInvOutput);
	
			//NAND
			resPullDown = CalculateOnResistance(widthNandN, NMOS, inputParameter.temperature, tech) * 2;
			tr = resPullDown * (capNandOutput + capInvInput);
			gm = CalculateTransconductance(widthNandN, NMOS, tech);
			beta = 1 / (resPullDown * gm);
			readLatency += horowitz(tr, beta, rampInvOutput, &rampNandOutput);
	
			//INV
			resPullDown = CalculateOnResistance(widthInvN, NMOS, inputParameter.temperature, tech);
			tr = resPullDown * (capInvOutput + capLoad);
			gm = CalculateTransconductance(widthInvN, NMOS, tech);
			beta = 1 / (resPullDown * gm);
			readLatency += horowitz(tr, beta, rampNandOutput, &rampInvOutput);
	
			readLatency *= numRead;
		}
		case Sigmoid:
		{
			cout << "[ActivationFunc] Error: Sigmoid is still under development!" 
				<< endl;
		}

		case Tanh:
		{
			cout << "[ActivationFunc] Error: Tanh is still under development!"
				<< endl;
		}

		default:
		{
			cout << "[ActivationFunc] Error: Please specify the activation function"
				<< endl;
		}
	}
}

void ActivationFunc::CalculatePower(double numRead) 
{
	if (!initialized) 
	{
		cout << "[ActivationFunc] Error: Require initialization first!" << endl;
		exit(-1);
	}

	leakage = 0;
	readDynamicEnergy = 0;

	// Leakage
	// INV
	leakage += CalculateGateLeakage(INV, 1, widthInvN, widthInvP, inputParameter.temperature, tech) * tech.vdd * 2; //For 2 Inv
	// NAND
	leakage += CalculateGateLeakage(NAND, 2, widthNandN, widthNandP, inputParameter.temperature, tech) * tech.vdd;

	// ReadDynamicEnergy
	// INV 1st
	readDynamicEnergy += (capInvOutput + capNandInput) * tech.vdd * tech.vdd * 2;
	// NAND
	readDynamicEnergy += (capInvInput + capNandOutput) * tech.vdd * tech.vdd;

	// INV 2nd
	readDynamicEnergy += (capInvOutput + capLoad) * tech.vdd * tech.vdd;

	// Overall
	leakage *= numCol * numIOBit * numRead;
	readDynamicEnergy *= numCol * numIOBit * numRead;
}

void ActivationFunc::PrintProperty(const char* str) {
	FunctionUnit::PrintProperty(str);
}

void ActivationFunc::SaveOutput(const char* str, const char* outputFile)
{
    FunctionUnit::SaveOutput(str, outputFile);
}
