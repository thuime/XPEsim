/*******************************************************************************
* Copyright (c) 2015-2017
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Shimeng Yu
* All rights reserved.
* 
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark 
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory). 
* Copyright of the model is maintained by the developers, and the model is distributed under 
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
* 
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer.
* 
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
* 
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
* Developer list: 
*   Pai-Yu Chen	    Email: pchen72 at asu dot edu 
*                    
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#include <cmath>
#include <iostream>
#include "constant.h"
#include "formula.h"
#include "LUT.h"

using namespace std;

LUT::LUT(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell): inputParameter(_inputParameter), tech(_tech), cell(_cell), mux(_inputParameter, _tech, _cell), muxDecoder(_inputParameter, _tech, _cell), colDecoder(_inputParameter, _tech, _cell), colDecoderDriver(_inputParameter, _tech, _cell), voltageSenseAmp(_inputParameter, _tech, _cell), FunctionUnit() {
	initialized = false;
}

void LUT::Initialize(bool _SRAM, int _numYbit, int _numEntry, double _clkFreq) {
	if (initialized)
		cout << "[LUT] Warning: Already initialized!" << endl;
	
	SRAM = _SRAM;
	numYbit = _numYbit;
	numEntry = _numEntry;
	clkFreq = _clkFreq;
	
	numCell = numYbit * numEntry;
	if (SRAM) {
		muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numEntry)), true, false);
		mux.Initialize(numYbit, numEntry, NULL, true);	// Digital Mux: treat it as FPGA type
	} else {
		colDecoder.Initialize(REGULAR_COL, (int)ceil(log2(numEntry)), false, false);
		colDecoderDriver.Initialize(COL_MODE, numCell, 1);
		muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numEntry)), true, false);
		double resTg = cell.resMemCellOn * IR_DROP_TOLERANCE;
		mux.Initialize(numYbit, numEntry, resTg, false);
		voltageSenseAmp.Initialize(numYbit, clkFreq);
	}

	// INV
	widthInvN = MIN_NMOS_SIZE * tech.featureSize;
	widthInvP = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;

	initialized = true;
}

void LUT::CalculateArea(double _newHeight, double _newWidth, AreaModify _option) {
	if (!initialized) {
		cout << "[LUT] Error: Require initialization first!" << endl;
	} else {
		double hInv, wInv;
		// INV
		CalculateGateArea(INV, 1, widthInvN, widthInvP, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &hInv, &wInv);

		double hUnit, wUnit;
		if (SRAM) {
			hUnit = hInv + cell.heightInFeatureSize * tech.featureSize;
			wUnit = MAX(wInv * 3, cell.widthInFeatureSize * tech.featureSize) * numYbit;
		} else {	// RRAM
			hUnit = cell.heightInFeatureSize * tech.featureSize;
			wUnit = cell.widthInFeatureSize * tech.featureSize * numYbit;
		}
		
		if (_newWidth && _option==NONE) {
			int numRowUnit;  // Number of rows of unit
			int numUnitPerRow;
			numUnitPerRow = (int)(_newWidth/wUnit);
			if (numUnitPerRow > numEntry) {
				numUnitPerRow = numEntry;
			}
			numRowUnit = (int)ceil((double)numEntry/numUnitPerRow);
			width = _newWidth;
			height = numRowUnit * hUnit;
		} else {
			width = numEntry * wUnit;
			height = hUnit;
		}
		
		// Get Mux height, compare it with Mux decoder height, and select whichever is larger for Mux
		mux.CalculateArea(NULL, width, NONE);
		muxDecoder.CalculateArea(NULL, NULL, NONE);
		double minMuxHeight = MAX(muxDecoder.height, mux.height);
		mux.CalculateArea(minMuxHeight, width, OVERRIDE);
		
		if (!SRAM) {	// RRAM
			voltageSenseAmp.CalculateUnitArea();
			voltageSenseAmp.CalculateArea(mux.widthTgShared);
		}

		height += minMuxHeight + voltageSenseAmp.height;
		width += muxDecoder.width;

		area = height * width;
		
		// Modify layout
		newHeight = _newHeight;
		newWidth = _newWidth;
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

void LUT::CalculateLatency(double _capLoad, double numRead) {
	if (!initialized) {
		cout << "[LUT] Error: Require initialization first!" << endl;
	} else {
		capLoad = _capLoad;
		readLatency = 0;

		double resCellAccess = CalculateOnResistance(cell.widthAccessCMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
		double capCellAccess = CalculateDrainCap(cell.widthAccessCMOS * tech.featureSize, NMOS, cell.widthInFeatureSize * tech.featureSize, tech);
		capSRAMCell = capCellAccess + CalculateDrainCap(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, cell.widthInFeatureSize * tech.featureSize, tech) + CalculateDrainCap(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, cell.widthInFeatureSize * tech.featureSize, tech);
		if (SRAM) {
			mux.CalculateLatency(1e20, capLoad, 1);
			muxDecoder.CalculateLatency(1e20, mux.capTgGateN*numYbit, mux.capTgGateP*numYbit, 1, 1);
			readLatency += mux.readLatency + muxDecoder.readLatency;
		} else {	// RRAM
			// Assuming no delay on RRAM wires
			mux.CalculateLatency(1e20, 0, 1);
			muxDecoder.CalculateLatency(1e20, mux.capTgGateN*numYbit, mux.capTgGateP*numYbit, 1, 1);
			double capInputLoad = mux.capTgDrain * (2 + numEntry - 1);
			voltageSenseAmp.CalculateLatency(capInputLoad, 1);
			readLatency += mux.readLatency + muxDecoder.readLatency + voltageSenseAmp.readLatency;
		}
		readLatency *= numRead;
	}
}

void LUT::CalculatePower(double numRead) {
	if (!initialized) {
		cout << "[LUT] Error: Require initialization first!" << endl;
	} else {
		leakage = 0;
		readDynamicEnergy = 0;

		if (SRAM) {
			mux.CalculatePower(1);
			muxDecoder.CalculatePower(1, 1);
			readDynamicEnergy += mux.readDynamicEnergy + muxDecoder.readDynamicEnergy;

			// Array leakage (assume 2 INV)
			leakage += CalculateGateLeakage(INV, 1, cell.widthSRAMCellNMOS * tech.featureSize,
					cell.widthSRAMCellPMOS * tech.featureSize, inputParameter.temperature, tech) * tech.vdd * 2;
			leakage *= numCell;

			leakage += mux.leakage;
			leakage += muxDecoder.leakage;
		} else {	// RRAM
			mux.CalculatePower(1);
			muxDecoder.CalculatePower(1, 1);
			voltageSenseAmp.CalculatePower(1);
			readDynamicEnergy += mux.readDynamicEnergy + muxDecoder.readDynamicEnergy + voltageSenseAmp.readDynamicEnergy;

			leakage += mux.leakage;
			leakage += muxDecoder.leakage;
			leakage += voltageSenseAmp.leakage;
		}
		readDynamicEnergy *= numRead;

		if (!readLatency) {
			//cout << "[LUT] Error: Need to calculate read latency first" << endl;
		} else {
			readPower = readDynamicEnergy/readLatency;
		}
	}
}

void LUT::PrintProperty(const char* str) {
	FunctionUnit::PrintProperty(str);
}

void LUT::SaveOutput(const char* str, const char* outputFile)
{
    FunctionUnit::SaveOutput(str, outputFile);
}