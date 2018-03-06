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
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include "constant.h"
#include "formula.h"
#include "SubArray.h"

using namespace std;

SubArray::SubArray(InputParameter& _inputParameter, Technology& _tech, MemCell& _cell):
						inputParameter(_inputParameter), tech(_tech), cell(_cell),
						wlDecoder(_inputParameter, _tech, _cell),
						wlDecoderOutput(_inputParameter, _tech, _cell),
						wlNewDecoderDriver(_inputParameter, _tech, _cell),
						wlNewSwitchMatrix(_inputParameter, _tech, _cell),
						rowCurrentSenseAmp(_inputParameter, _tech, _cell),
						parallelCurrentSenseAmp(_inputParameter, _tech, _cell),
						mux(_inputParameter, _tech, _cell),
						muxDecoder(_inputParameter, _tech, _cell),
						slSwitchMatrix(_inputParameter, _tech, _cell),
						blSwitchMatrix(_inputParameter, _tech, _cell),
						wlSwitchMatrix(_inputParameter, _tech, _cell),
						deMux(_inputParameter, _tech, _cell),
						readCircuit(_inputParameter, _tech, _cell),
						voltageSenseAmp(_inputParameter, _tech, _cell),
						precharger(_inputParameter, _tech, _cell),
						senseAmp(_inputParameter, _tech, _cell),
						colDecoder(_inputParameter, _tech, _cell),
						wlDecoderDriver(_inputParameter, _tech, _cell),
						colDecoderDriver(_inputParameter, _tech, _cell),
						sramWriteDriver(_inputParameter, _tech, _cell),
						dff(_inputParameter, _tech, _cell),
						adder(_inputParameter, _tech, _cell),
						shiftAdd(_inputParameter, _tech, _cell),
						intermux(_inputParameter, _tech, _cell),
						intermuxDecoder(_inputParameter, _tech, _cell),
						interdff(_inputParameter, _tech, _cell),
                        comparator(_inputParameter, _tech, _cell),
						multilevelSenseAmp(_inputParameter, _tech, _cell),
						multilevelSAEncoder(_inputParameter, _tech, _cell){
	initialized = false;
	readDynamicEnergyArray = writeDynamicEnergyArray = 0;
}

void SubArray::Initialize(int _numRow, int _numCol, double _unitWireRes){  //initialization module
	if (initialized)
		cout << "[Subarray] Warning: Already initialized!" << endl;  //avioding initialize twice
	
	numRow = _numRow;    //import parameters
	numCol = _numCol;
	unitWireRes = _unitWireRes;
	
	double MIN_CELL_HEIGHT = MAX_TRANSISTOR_HEIGHT;  //set real layout cell height
	double MIN_CELL_WIDTH = (MIN_GAP_BET_GATE_POLY + POLY_WIDTH) * 2;  //set real layout cell width
	if (cell.memCellType == Type::SRAM) {  //if array is SRAM
		// SRAM not support for TsingHua 
	
	} else if (cell.memCellType == Type::RRAM) {  //if array is RRAM
		double cellHeight = cell.heightInFeatureSize * sqrt(cell.multipleCells);  //set RRAM cell height, for multipleCells, cell number is N^2, eg: cell number is 9, means the cells are arranged as 3*3, so height is F*3
		double cellWidth = cell.widthInFeatureSize * sqrt(cell.multipleCells);  //set RRAM cell width
		if (cell.accessType == CMOS_access) {  // 1T1R
			if (relaxArrayCellWidth) {
				lengthRow = (double)numCol * MAX(cellWidth, MIN_CELL_WIDTH*2) * tech.featureSize;	// Width*2 because generally switch matrix has 2 pass gates per column, even the SL/BL driver has 2 pass gates per column in traditional 1T1R memory
			} else {
				lengthRow = (double)numCol * cellWidth * tech.featureSize;
			}
			if (relaxArrayCellHeight) {
				lengthCol = (double)numRow * MAX(cellHeight, MIN_CELL_HEIGHT) * tech.featureSize;
			} else {
				lengthCol = (double)numRow * cellHeight * tech.featureSize;
			}
		} else {	// Cross-point, if enter anything else except 'CMOS_access'
			if (relaxArrayCellWidth) {
				lengthRow = (double)numCol * MAX(cellWidth* tech.featureSize, MIN_CELL_WIDTH*2*tech.featureSize);	// Width*2 because generally switch matrix has 2 pass gates per column, even the SL/BL driver has 2 pass gates per column in traditional 1T1R memory
			} else {
				lengthRow = (double)numCol * cellWidth * tech.featureSize;
			}
			if (relaxArrayCellHeight) {
				lengthCol = (double)numRow * MAX(cellHeight* tech.featureSize, MIN_CELL_HEIGHT*tech.featureSize);
			} else {  
				lengthCol = (double)numRow * cellHeight * tech.featureSize;
			}
		}
	}      //finish setting array size
	
	capRow1 = lengthRow * 0.2e-15/1e-6;	// BL for 1T1R, WL for Cross-point and SRAM
	capRow2 = lengthRow * 0.2e-15/1e-6;	// WL for 1T1R
	capCol = lengthCol * 0.2e-15/1e-6;
	
	resRow = lengthRow * unitWireRes; 
	resCol = lengthCol * unitWireRes;
	

	//start to initializing the subarray modules
	if (cell.memCellType == Type::SRAM) {  //if array is SRAM
		
		//SRAM not support for TsingHua 

    } else if (cell.memCellType == Type::RRAM) {
		if (cell.accessType == CMOS_access) {	// 1T1R
			cell.resCellAccess = cell.resistanceOn * IR_DROP_TOLERANCE;    //calculate access CMOS resistance
			cell.widthAccessCMOS = CalculateOnResistance(tech.featureSize, NMOS, inputParameter.temperature, tech) / cell.resCellAccess;   //get access CMOS width
			if (cell.widthAccessCMOS > cell.widthInFeatureSize) {	// Place transistor vertically
				printf("Transistor width of 1T1R=%.2fF is larger than the assigned cell width=%.2fF in layout\n", cell.widthAccessCMOS, cell.widthInFeatureSize);
				exit(-1);
			}

			cell.resMemCellOn = cell.resCellAccess + cell.resistanceOn;       //calculate single memory cell resistance_ON
			cell.resMemCellOff = cell.resCellAccess + cell.resistanceOff;      //calculate single memory cell resistance_OFF
			cell.resMemCellAvg = cell.resCellAccess + cell.resistanceAvg;      //calculate single memory cell resistance_AVG

			capRow2 += CalculateGateCap(cell.widthAccessCMOS * tech.featureSize, tech) * numCol;          //sum up all the gate cap of access CMOS, as the row cap
			capCol += CalculateDrainCap(cell.widthAccessCMOS * tech.featureSize, NMOS, cell.widthInFeatureSize * tech.featureSize, tech) * numRow;	// If capCol is found to be too large, increase cell.widthInFeatureSize to relax the limit

			if (FPGA) {	
				// TODO
				cout << "Currently RRAM does not support FPGA mode" << endl;

			} else if (!multifunctional && !neuro) {	// Memory only (regular 1T1R)
				
			} else {	// Neuro included (pseudo-crossbar 1T1R)
				if (digitalModeNeuro) {  //if digital mode pseudo-1T1R
					
				} 
				else if (newBNNrowbyrowMode) {
					// row-by-row BNN
				}
				else if (newBNNparallelMode) {
					// Parallel BNN
				}
				else if (XNORModeDoubleEnded) {
					// XNOR within double-ended S/A
				}
				else if (XNORModeSingleEnded) {
					// XNOR within single-ended S/A
				} else if (tsqinghua) {   // tsqinghua's design ... multiple analog cells as one synapses
				    double resTg = cell.resMemCellOn / numRow * IR_DROP_TOLERANCE;
					wlNewSwitchMatrix.Initialize(numRow, clkFreq);         // initialize new WL Switch Matrix
					
					slSwitchMatrix.Initialize(COL_MODE, numCol, resTg*numRow, neuro, parallelWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, numWritePulse, clkFreq);     //SL use switch matrix
					
					int numInput = (int)ceil((double)numCol/numColMuxed); //input number of mux (num of column/ num of column that share one SA)
					mux.Initialize(numInput, numColMuxed, resTg, FPGA);

					if (numColMuxed > 1) {    //if more than one column share one SA
						muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numColMuxed)), true, false);
					}					
					multilevelSenseAmp.Initialize(numCol, levelOutput, clkFreq, numReadCellPerOperationNeuro);
					multilevelSAEncoder.Initialize(levelOutput, numCol);
					
				} else {  //analog mode pesudo-1T1R

				}
			}

		} else {	// Cross-point
			
			// The nonlinearity is from the selector, assuming RRAM itself is linear
			if (cell.nonlinearIV) {   //introduce nonlinearity to the RRAM resistance
				cell.resMemCellOn = cell.resistanceOn;
				cell.resMemCellOff = cell.resistanceOff;
				cell.resMemCellOnAtHalfVw = NonlinearResistance(cell.resistanceOn, cell.nonlinearity, cell.writeVoltage, cell.readVoltage, cell.writeVoltage/2);
				cell.resMemCellOffAtHalfVw = NonlinearResistance(cell.resistanceOff, cell.nonlinearity, cell.writeVoltage, cell.readVoltage, cell.writeVoltage/2);
				cell.resMemCellOnAtVw = NonlinearResistance(cell.resistanceOn, cell.nonlinearity, cell.writeVoltage, cell.readVoltage, cell.writeVoltage);
				cell.resMemCellOffAtVw = NonlinearResistance(cell.resistanceOff, cell.nonlinearity, cell.writeVoltage, cell.readVoltage, cell.writeVoltage);
				cell.resMemCellAvg = cell.resistanceAvg;
				cell.resMemCellAvgAtHalfVw = (cell.resMemCellOnAtHalfVw + cell.resMemCellOffAtHalfVw) / 2;
				cell.resMemCellAvgAtVw = (cell.resMemCellOnAtVw + cell.resMemCellOffAtVw) / 2;
			} else {  //simply assume RRAM resistance is linear
				cell.resMemCellOn = cell.resistanceOn;
				cell.resMemCellOff = cell.resistanceOff;
				cell.resMemCellOnAtHalfVw = cell.resistanceOn;
				cell.resMemCellOffAtHalfVw = cell.resistanceOff;
				cell.resMemCellOnAtVw = cell.resistanceOn;
				cell.resMemCellOffAtVw = cell.resistanceOff;
				cell.resMemCellAvg = cell.resistanceAvg;
				cell.resMemCellAvgAtHalfVw = cell.resistanceAvg;
				cell.resMemCellAvgAtVw = cell.resistanceAvg;
			}

			if (FPGA) {
				// TODO
				cout << "Currently RRAM does not support FPGA mode" << endl;

			} else if (!multifunctional && !neuro) {   // Memory only
				
			} else {    // Neuro included
				if (digitalModeNeuro) {  //digital mode cross-point, need wlDecoder, colDecoder and colDecoderDriver, MUX and VSA, Adder, ShiftAdd and DFF
					
				} else if (tsqinghua) {   // tsqinghua's design ... multiple analog cells as one synapses
				    double resTg = cell.resMemCellOnAtVw / numCol * IR_DROP_TOLERANCE;
					wlSwitchMatrix.Initialize(ROW_MODE, numRow, resTg, neuro, parallelWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, numWritePulse, clkFreq);
					
					resTg = cell.resMemCellOnAtVw / numRow * IR_DROP_TOLERANCE;
					blSwitchMatrix.Initialize(COL_MODE, numCol, resTg, neuro, parallelWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, numWritePulse, clkFreq);

					int numInput = (int)ceil((double)numCol/numColMuxed);
					resTg = cell.resMemCellOnAtVw / numRow * IR_DROP_TOLERANCE;
					mux.Initialize(numInput, numColMuxed, resTg, FPGA);

					if (numColMuxed > 1) {
						muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numColMuxed)), true, false);
					}
					
					multilevelSenseAmp.Initialize(numCol, levelOutput, clkFreq, numReadCellPerOperationNeuro);
					multilevelSAEncoder.Initialize(levelOutput, numCol);
				} else {  //analog mode cross-point
					
				}
			}
		}
	}


	initialized = true;  //finish initialization
}



void SubArray::CalculateArea() {  //calculate layout area for total design
	if (!initialized) {
		cout << "[Subarray] Error: Require initialization first!" << endl;  //ensure initialization first
	} else {  //if initialized, start to do calculation
		if (cell.memCellType == Type::SRAM) {       
			
            // SRAM not support for TsingHua

	    } else if (cell.memCellType == Type::RRAM) {
			if (cell.accessType == CMOS_access) {	// 1T1R
				
				// Array only
				heightArray = lengthCol;
				widthArray = lengthRow;
				areaArray = heightArray * widthArray;
				
				if (FPGA) {
					// TODO
					cout << "Currently RRAM does not support FPGA mode" << endl;

				} else if (!multifunctional && !neuro) {   // Memory only

				} else {    // Neuro included
					if (digitalModeNeuro) {
						// original digital 
					} 
					else if (newBNNrowbyrowMode) {
						// row-by-row BNN
				    }
				    else if (newBNNparallelMode) {
					    // parallel BNN
				    }
					else if (XNORModeDoubleEnded) {
						// XNOR within double-ended S/A
					}
					else if (XNORModeSingleEnded) {
						// XNOR within single-ended S/A
					} else if (tsqinghua) {   // tsqinghua's design ... multiple analog cells as one synapses
					    wlNewSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
					    slSwitchMatrix.CalculateArea(NULL, widthArray, MAGIC);
						// Get Mux height, compare it with Mux decoder height, and select whichever is larger for Mux
						mux.CalculateArea(NULL, widthArray, NONE);
						muxDecoder.CalculateArea(NULL, NULL, NONE);
						double minMuxHeight = MAX(muxDecoder.height, mux.height);
						mux.CalculateArea(minMuxHeight, widthArray, OVERRIDE);
					    multilevelSenseAmp.CalculateArea(NULL, widthArray, NONE);
						multilevelSAEncoder.CalculateArea(NULL, widthArray, NONE);
						
						height = slSwitchMatrix.height + heightArray + mux.height + multilevelSenseAmp.height + multilevelSAEncoder.height;
						width = MAX(wlNewSwitchMatrix.width, muxDecoder.width) + widthArray;
						area = height * width;
						usedArea = areaArray + wlDecoder.area + wlNewSwitchMatrix.area + slSwitchMatrix.area + mux.area + multilevelSenseAmp.area + muxDecoder.area + multilevelSAEncoder.area;
						emptyArea = area - usedArea;
						
					} else {  //analog mode 1T1R RRAM
						
					}
				}

			} else {        // Cross-point
				
				// Array only
				heightArray = lengthCol;
				widthArray = lengthRow;
				areaArray = heightArray * widthArray;
				
				if (FPGA) {
					// TODO
					cout << "Currently RRAM does not support FPGA mode" << endl;

				} else if (!multifunctional && !neuro) {   // Memory only
					
				} else {    // Neuro included
					if (digitalModeNeuro) {  //digital cross-point 

					} else if (tsqinghua) {   // tsqinghua's design ... multiple analog cells as one synapses
					    wlSwitchMatrix.CalculateArea(heightArray, NULL, MAGIC);
						blSwitchMatrix.CalculateArea(NULL, widthArray, MAGIC);
					    mux.CalculateArea(NULL, widthArray, NONE);
						muxDecoder.CalculateArea(NULL, NULL, NONE);
						double minMuxHeight = MAX(muxDecoder.height, mux.height);
						mux.CalculateArea(minMuxHeight, widthArray, OVERRIDE);
					    multilevelSenseAmp.CalculateArea(NULL, widthArray, NONE);
						multilevelSAEncoder.CalculateArea(NULL, widthArray, NONE);
						
						height = blSwitchMatrix.height + heightArray + mux.height + multilevelSenseAmp.height + multilevelSAEncoder.height;
						width = MAX(wlSwitchMatrix.width, muxDecoder.width) + widthArray;
						area = height * width;
						usedArea = areaArray + wlSwitchMatrix.area + blSwitchMatrix.area + mux.area + muxDecoder.area + multilevelSenseAmp.area + multilevelSAEncoder.area;
						emptyArea = area - usedArea;
						
					} else {  //analog cross-point
						
					}
				}
			}
		}
	}
}

void SubArray::CalculateLatency(double _rampInput) {   //calculate latency for different mode 
	if (!initialized) {
		cout << "[Subarray] Error: Require initialization first!" << endl;
	} else {
		
		if (cell.memCellType == Type::SRAM) {
			
			// SRAM not support for TsingHua
			
	    } else if (cell.memCellType == Type::RRAM) {
			if (cell.accessType == CMOS_access) {   // 1T1R
				
				if (FPGA) {
					// TODO
					cout << "Currently RRAM does not support FPGA mode" << endl;

				} else if (!neuro) {	// Memory mode
					if (multifunctional) {	// Multifunctional architecture
						
					} else {	// Memory only
						
					}
				} else {	// Neuro mode
					if (digitalModeNeuro) {
						
					} 
					else if (newBNNrowbyrowMode) {
						// row-by-row BNN
				    }
				    else if (newBNNparallelMode) {
						// parallel BNN
				    }
					else if (XNORModeDoubleEnded) {
						// XNOR within double-ended S/A
					}
					else if (XNORModeSingleEnded) {
						// XNOR within single-ended S/A
					} else if (tsqinghua) {   // tsqinghua's design ... multiple analog cells as one synapses
					    double capBL = lengthCol * 0.2e-15/1e-6;
						int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
					    wlNewSwitchMatrix.CalculateLatency(1e20, capRow2, resRow, numColMuxed/2, 2*numWriteOperationPerRow*numRow*activityRowWrite);    // 2 means 2-step write
					    slSwitchMatrix.CalculateLatency(1e20, capCol, resCol, 1, 2*numWriteOperationPerRow*numRow*activityRowWrite);

						// Calculate column latency
						double colRamp = 0;
						double tau = resCol * capCol / 2 * (cell.resMemCellOff + resCol / 3) / (cell.resMemCellOff + resCol);
						colDelay = horowitz(tau, 0, blSwitchMatrix.rampOutput, &colRamp);	// Just to generate colRamp

						mux.CalculateLatency(colRamp, 0, 1);
						int numInput = (int)ceil((double)numCol/numColMuxed);
						muxDecoder.CalculateLatency(1e20, mux.capTgGateN*numInput, mux.capTgGateP*numInput, 1, 1);
						
					    multilevelSenseAmp.CalculateLatency(numColMuxed*numReadPulse);
						multilevelSAEncoder.CalculateLatency(1e20, numColMuxed*numReadPulse);
						
						readLatency = 0;
						readLatency += MAX(wlNewSwitchMatrix.readLatency, muxDecoder.readLatency + muxDecoder.readLatency);
						readLatency += multilevelSenseAmp.readLatency;
						readLatency += multilevelSAEncoder.readLatency;

						// Write
						// Write
						writeLatency = 0;
						writeLatency += wlNewSwitchMatrix.writeLatency;
						writeLatency += slSwitchMatrix.writeLatency;
						
					} else {
						
					}
				}

			} else {	// Cross-point
				if (FPGA) {
					// TODO
					cout << "Currently RRAM does not support FPGA mode" << endl;

				} else if (!neuro) {	// Memory mode
					if (multifunctional) {  // Multifunctional architecture
						
					} else {	// Memory only
						
					}

				} else {	// Neuro mode
					if (digitalModeNeuro) {
						
					} else if (tsqinghua) {   // tsqinghua's design ... multiple analog cells as one synapses
					    int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
						if (parallelWrite) {    // parallel write
							wlSwitchMatrix.CalculateLatency(1e20, capRow1, resRow, numReadPulse, maxNumWritePulse*2);    // R>0 and R<0 phase
						} else {    // row-by-row write
							wlSwitchMatrix.CalculateLatency(1e20, capRow1, resRow, numReadPulse, maxNumWritePulse*2*numWriteOperationPerRow*numRow*activityRowWrite);	// *2 means 2-step write
						}
						blSwitchMatrix.CalculateLatency(1e20, capCol, resCol, 1, 1);

						// Calculate column latency
						double colRamp = 0;
						double tau = resCol * capCol / 2 * (cell.resMemCellOff + resCol / 3) / (cell.resMemCellOff + resCol);
						colDelay = horowitz(tau, 0, wlSwitchMatrix.rampOutput, &colRamp);	// Just to generate colRamp

						mux.CalculateLatency(colRamp, 0, 1);
						int numInput = (int)ceil((double)numCol/numColMuxed);
						muxDecoder.CalculateLatency(1e20, mux.capTgGateN*numInput, mux.capTgGateP*numInput, 1, 1);
						
					    multilevelSenseAmp.CalculateLatency(numColMuxed*numReadPulse);
						multilevelSAEncoder.CalculateLatency(1e20, numColMuxed*numReadPulse);
						
						readLatency = 0;
						readLatency += MAX(wlSwitchMatrix.readLatency, muxDecoder.readLatency + muxDecoder.readLatency);
						readLatency += multilevelSenseAmp.readLatency; 
                        readLatency += multilevelSAEncoder.readLatency;						

						// Write
						writeLatency = 0;
						writeLatency += MAX(wlSwitchMatrix.writeLatency, blSwitchMatrix.writeLatency);
						
					} else {
						// analog crossbar
					}
				}

			}
		}
	}
}

void SubArray::CalculatePower(double activityRowRead) {
	if (!initialized) {
		cout << "[Subarray] Error: Require initialization first!" << endl;
	} else {
		
		if (cell.memCellType == Type::SRAM) {
			
			// SRAM not support for TsingHua
			
	    } else if (cell.memCellType == Type::RRAM) {
			if (cell.accessType == CMOS_access) {   // 1T1R
							
				if (FPGA) {
					// TODO
					cout << "Currently RRAM does not support FPGA mode" << endl;

				} else if (!neuro) {	// Memory mode
					if (multifunctional) {  // Multifunctional architecture
						
					} else {	// Memory only
						
					}

				} else {	// Neuro mode
					if (digitalModeNeuro) {
						// original digital
					} 
					else if (newBNNrowbyrowMode) {
						// row-by-row BNN
				    }
				    else if (newBNNparallelMode) {
						// parallel BNN
				    }
					else if (XNORModeDoubleEnded) {
						// XNOR within double-ended S/A
					}
					else if (XNORModeSingleEnded) {
						// XNOR within single-ended S/A
					} else if (tsqinghua) {   // tsqinghua's design ... multiple analog cells as one synapses
						double numReadCells = (int)ceil((double)numCol/numColMuxed);    // similar parameter as numReadCellPerOperationNeuro, which is for SRAM
						int numWriteCells = (int)ceil((double)numCol/numWriteColMuxed);
					    wlNewSwitchMatrix.CalculatePower(activityRowRead*numReadPulse*numColMuxed, numRow*activityRowWrite, activityRowRead);    // 2 means 2-step write
						
						double numWriteOperationPerRow;   // average value (can be non-integer for energy calculation)
						if (numCol * activityColWrite > numWriteCellPerOperationNeuro)
							numWriteOperationPerRow = numCol * activityColWrite / numWriteCellPerOperationNeuro;
						else
							numWriteOperationPerRow = 1;
					    slSwitchMatrix.CalculatePower(1, numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
						
						mux.CalculatePower(numColMuxed*numReadPulse);
						muxDecoder.CalculatePower(numColMuxed*numReadPulse, 1);
						multilevelSenseAmp.CalculatePower(numof1, numof2, numof3, numof4, numof5, numof6, numof7, numof8, numof9, numof10, numColMuxed*numReadPulse);
						multilevelSAEncoder.CalculatePower(numColMuxed*numReadPulse);
							
						// Array (here it is regular 1T1R array)
						double capBL = lengthCol * 0.2e-15/1e-6;
						readDynamicEnergyArray = 0;
						readDynamicEnergyArray += capBL * cell.readVoltage * cell.readVoltage * numReadCells; // Selected BLs
						readDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd; // Selected WL
						readDynamicEnergyArray *= numRow * activityRowRead * numReadPulse * numColMuxed;
						
						// SET
						writeDynamicEnergyArray = 0;
						writeDynamicEnergyArray += capBL * cell.writeVoltage * cell.writeVoltage * MIN(numWriteCells, numCol)/2;
						writeDynamicEnergyArray += cell.writeVoltage * cell.writeVoltage * (1/cell.resMemCellOn + 1/cell.resMemCellOff) / 2 * MIN(numWriteCells, numCol) / 2 * cell.writePulseWidth; // half SET with Ron and Roff 50/50
						// RESET
						writeDynamicEnergyArray += capCol * cell.writeVoltage * cell.writeVoltage * MIN(numWriteCells, numCol)/2;
						writeDynamicEnergyArray += cell.writeVoltage * cell.writeVoltage * (1/cell.resMemCellOn + 1/cell.resMemCellOff) / 2 * MIN(numWriteCells, numCol) / 2 * cell.writePulseWidth;    // half RESET with Ron and Roff 50/50
						writeDynamicEnergyArray *= numRow * activityRowWrite * numWriteColMuxed;
						//cout << writeDynamicEnergyArray << "writeDynamicEnergyArray"<< endl;
						
						// Read
						readDynamicEnergy = 0;
						readDynamicEnergy += readDynamicEnergyArray;
						readDynamicEnergy += wlNewSwitchMatrix.readDynamicEnergy;
						readDynamicEnergy += mux.readDynamicEnergy;
						readDynamicEnergy += muxDecoder.readDynamicEnergy;
						readDynamicEnergy += parallelCurrentSenseAmp.readDynamicEnergy;
						readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy;
						readDynamicEnergy += multilevelSAEncoder.readDynamicEnergy;

						// Write
						writeDynamicEnergy = 0;
						writeDynamicEnergy += writeDynamicEnergyArray;
						writeDynamicEnergy += wlNewSwitchMatrix.writeDynamicEnergy;
						writeDynamicEnergy += slSwitchMatrix.writeDynamicEnergy;
						writeDynamicEnergy += writeDynamicEnergyArray;
						
				    } else {
						// analog 1T1R
					}
				}
				
				// Leakage
				leakage = 0;
				leakage += wlDecoder.leakage;
				leakage += wlDecoderOutput.leakage;
				leakage += colDecoder.leakage;
				leakage += colDecoderDriver.leakage;
				leakage += blSwitchMatrix.leakage;
				leakage += slSwitchMatrix.leakage;
				leakage += mux.leakage;
				leakage += muxDecoder.leakage;
				leakage += deMux.leakage;
				leakage += readCircuit.leakage;
				leakage += voltageSenseAmp.leakage;
				leakage += shiftAdd.leakage;

			} else {	// Cross-point

				if (FPGA) {
					// TODO
					cout << "Currently RRAM does not support FPGA mode" << endl;

				} else if (!neuro) {    // Memory mode
					if (multifunctional) {  // Multifunctional architecture
						
					} else {	// Memory only
						
					}

				} else {    // Neuro mode
					if (parallelWrite) {	// parallel write	// FIXME

					} else {	// row-by-row write
						if (digitalModeNeuro) {
							
						} else if (tsqinghua) {   // tsqinghua's design ... multiple analog cells as one synapses
					        double numWriteOperationPerRow;   // average value (can be non-integer for energy calculation)
							if (numCol * activityColWrite > numWriteCellPerOperationNeuro)
								numWriteOperationPerRow = numCol * activityColWrite / numWriteCellPerOperationNeuro;
							else
								numWriteOperationPerRow = 1;
							
							wlSwitchMatrix.CalculatePower(numReadPulse, numRow*activityRowWrite, activityRowRead, activityColWrite);
							blSwitchMatrix.CalculatePower(1, numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
							mux.CalculatePower(numColMuxed*numReadPulse);
							muxDecoder.CalculatePower(numColMuxed*numReadPulse, 1);
							multilevelSenseAmp.CalculatePower(numof1, numof2, numof3, numof4, numof5, numof6, numof7, numof8, numof9, numof10, numColMuxed*numReadPulse);
						    multilevelSAEncoder.CalculatePower(numColMuxed*numReadPulse);
	
							// Array
							readDynamicEnergyArray = 0;
							readDynamicEnergyArray += capRow1 * cell.readVoltage * cell.readVoltage * numRow * (1-activityRowRead);    // unselected WLs
							readDynamicEnergyArray += capCol * cell.readVoltage * cell.readVoltage * numCol;    // read all columns in total 
							readDynamicEnergyArray *= numReadPulse;
							
							// Use average case in write energy calculation: half LTP and half LTD with average resistance
							double totalWriteTime = cell.writePulseWidth * maxNumWritePulse;
							// LTP
							writeDynamicEnergyArray = 0;
							writeDynamicEnergyArray += capRow1 * cell.writeVoltage * cell.writeVoltage;	// Selected WL
							writeDynamicEnergyArray += capCol * cell.writeVoltage * cell.writeVoltage * MIN(numWriteCellPerOperationNeuro, numCol*activityColWrite) / 2 * numWritePulse;	// Selected BLs
							writeDynamicEnergyArray += cell.writeVoltage * cell.writeVoltage / cell.resMemCellAvgAtVw * MIN(numWriteCellPerOperationNeuro, numCol*activityColWrite) / 2 * numWritePulse * cell.writePulseWidth;	// LTP
							writeDynamicEnergyArray += cell.writeVoltage/2 * cell.writeVoltage/2 / cell.resMemCellAvgAtHalfVw * (numCol - MIN(numWriteCellPerOperationNeuro, numCol*activityColWrite)/2) * totalWriteTime;    // Half-selected cells on the row
							writeDynamicEnergyArray += cell.writeVoltage/2 * cell.writeVoltage/2 / cell.resMemCellAvgAtHalfVw * (numRow-1) * MIN(numWriteCellPerOperationNeuro, numCol*activityColWrite) / 2 * totalWriteTime;   // Half-selected cells on the selected columns
							// LTD
							writeDynamicEnergyArray += capCol * cell.writeVoltage * cell.writeVoltage * MIN(numWriteCellPerOperationNeuro, numCol*activityColWrite) / 2 * numWritePulse;  // Selected BLs
							writeDynamicEnergyArray += cell.writeVoltage * cell.writeVoltage / cell.resMemCellAvgAtVw * MIN(numWriteCellPerOperationNeuro, numCol*activityColWrite) / 2 * numWritePulse * cell.writePulseWidth; // LTD
							writeDynamicEnergyArray += cell.writeVoltage/2 * cell.writeVoltage/2 / cell.resMemCellAvgAtHalfVw * (numCol - MIN(numWriteCellPerOperationNeuro, numCol*activityColWrite)/2) * totalWriteTime;    // Half-selected cells on the row
							writeDynamicEnergyArray += cell.writeVoltage/2 * cell.writeVoltage/2 / cell.resMemCellAvgAtHalfVw * (numRow-1) * MIN(numWriteCellPerOperationNeuro, numCol*activityColWrite) / 2 * totalWriteTime;   // Half-selected cells on the selected columns
							// Both SET and RESET
							writeDynamicEnergyArray += capCol * cell.writeVoltage/2 * cell.writeVoltage/2 * numCol; // Unselected BLs (every BL has one time to charge to V/2 within the 2-step write)
							writeDynamicEnergyArray += capRow1 * cell.writeVoltage/2 * cell.writeVoltage/2 * (numRow-1);  // Unselected WLs
							
							writeDynamicEnergyArray *= numWriteOperationPerRow * numRow * activityRowWrite;
							
							// Read
							readDynamicEnergy = 0;
							readDynamicEnergy += wlSwitchMatrix.readDynamicEnergy;
							readDynamicEnergy += mux.readDynamicEnergy;
							readDynamicEnergy += muxDecoder.readDynamicEnergy;
							readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy;
							readDynamicEnergy += multilevelSAEncoder.readDynamicEnergy;

							// Write
							writeDynamicEnergy = 0;
							writeDynamicEnergy += wlSwitchMatrix.writeDynamicEnergy;
							writeDynamicEnergy += blSwitchMatrix.writeDynamicEnergy;
							writeDynamicEnergy += writeDynamicEnergyArray;
						
					    } else {
							// analog crossbar
						}
					}
				}
				
				// Leakage
				leakage = 0;
				leakage += wlDecoder.leakage;
				leakage += wlDecoderDriver.leakage;
				leakage += colDecoder.leakage;
				leakage += colDecoderDriver.leakage;
				leakage += wlSwitchMatrix.leakage;
				leakage += blSwitchMatrix.leakage;
				leakage += mux.leakage;
				leakage += muxDecoder.leakage;
				leakage += deMux.leakage;
				leakage += readCircuit.leakage;
				leakage += voltageSenseAmp.leakage;
				leakage += shiftAdd.leakage;

			}
		}

		if (!readLatency) {
			cout << "[SubArray] Error: Need to calculate read latency first" << endl;
		} else {
			readPower = readDynamicEnergy/readLatency + leakage;
		}
		if (!writeLatency) {
			cout << "[SubArray] Error: Need to calculate write latency first" << endl;
		} else {
			writePower = writeDynamicEnergy/writeLatency + leakage;
		}

	}
}

void SubArray::PrintProperty() {
	cout << endl;
	cout << "Array:" << endl;
	cout << "Area = " << heightArray*1e6 << "um x " << widthArray*1e6 << "um = " << areaArray*1e12 << "um^2" << endl;
	cout << "Read Dynamic Energy = " << readDynamicEnergyArray*1e12 << "pJ" << endl;
	cout << "Write Dynamic Energy = " << writeDynamicEnergyArray*1e12 << "pJ" << endl;
	if (cell.memCellType == Type::SRAM) {
		// SRAM not support for TsingHua
	} else if (cell.memCellType == Type::RRAM) {
		if (cell.accessType == CMOS_access) {   // 1T1R
			if (FPGA) {
				// TODO
				cout << "Currently RRAM does not support FPGA mode" << endl;

			} else if (!multifunctional && !neuro) {   // Memory only (regular 1T1R)
				
			} else {	// Neuro
				if (digitalModeNeuro) {
					// original digital
				} 
				else if (newBNNrowbyrowMode) {
					// row-by-row BNN
				}
				else if (XNORModeDoubleEnded) {
					// parallel BNN
				}
				else if (XNORModeSingleEnded) {
					// XNOR within single-ended S/A
				}
				else if (newBNNparallelMode) {
					// XNOR within double-ended S/A
				} else if (tsqinghua) {
					wlNewSwitchMatrix.PrintProperty("wlNewSwitchMatrix");
					slSwitchMatrix.PrintProperty("slSwitchMatrix");
					mux.PrintProperty("mux");
					muxDecoder.PrintProperty("muxDecoder");
					multilevelSenseAmp.PrintProperty("multilevelSenseAmp");
					multilevelSAEncoder.PrintProperty("multilevelSAEncoder");
				} else {
					if (multifunctional) {  // Multifunctional architecture
						
					} else {	// Neuro only
						
					}
				}
			}
		}
		else {	// Crosspoint
			if (FPGA) {
				// TODO
				cout << "Currently RRAM does not support FPGA mode" << endl;

			} else if (!multifunctional && !neuro) {   // Memory only
				
			} else {    // Neuro included
				if (digitalModeNeuro) {
					
				} else if (tsqinghua) {
					wlSwitchMatrix.PrintProperty("wlSwitchMatrix");
					blSwitchMatrix.PrintProperty("blSwitchMatrix");
					mux.PrintProperty("mux");
					muxDecoder.PrintProperty("muxDecoder");
					multilevelSenseAmp.PrintProperty("multilevelSenseAmp");
					multilevelSAEncoder.PrintProperty("multilevelSAEncoder");
				} else {
					if (multifunctional) {  // Multifunctional architecture
						
					} else {    // Neuro only
						
					}
				}
			}
		}
	}
	FunctionUnit::PrintProperty("Synaptic Core");
	cout << "Used Area = " << usedArea*1e12 << "um^2" << endl;
	cout << "Empty Area = " << emptyArea*1e12 << "um^2" << endl;
}

void SubArray::SaveOutput(const char* outputFile) {
	ofstream outfile;                                           
	outfile.open(outputFile, ios::app); 
	outfile << endl;
	outfile << "---------------------------------------------------------" << endl;
	outfile << endl;
	outfile << "Array:" << endl;
	outfile << "Area = " << heightArray*1e6 << "um x " << widthArray*1e6 << "um = " << areaArray*1e12 << "um^2" << endl;
	outfile << "Read Dynamic Energy = " << readDynamicEnergyArray*1e12 << "pJ" << endl;
	outfile << "Write Dynamic Energy = " << writeDynamicEnergyArray*1e12 << "pJ" << endl;
	
	if (cell.accessType == CMOS_access) {    // 1T1R
		wlNewSwitchMatrix.SaveOutput("wlNewSwitchMatrix", outputFile);
		slSwitchMatrix.SaveOutput("slSwitchMatrix", outputFile);
		mux.SaveOutput("mux", outputFile);
		muxDecoder.SaveOutput("muxDecoder", outputFile);
		multilevelSenseAmp.SaveOutput("multilevelSenseAmp", outputFile);
		multilevelSAEncoder.SaveOutput("multilevelSAEncoder", outputFile);
	} else {                                 // crossbar
		wlSwitchMatrix.SaveOutput("wlSwitchMatrix", outputFile);
		blSwitchMatrix.SaveOutput("blSwitchMatrix", outputFile);
		mux.SaveOutput("mux", outputFile);
		muxDecoder.SaveOutput("muxDecoder", outputFile);
		multilevelSenseAmp.SaveOutput("multilevelSenseAmp", outputFile);
		multilevelSAEncoder.SaveOutput("multilevelSAEncoder", outputFile);
	}
	FunctionUnit::SaveOutput("Synaptic Core", outputFile);
	outfile << "Used Area = " << usedArea*1e12 << "um^2" << endl;
	outfile << "Empty Area = " << emptyArea*1e12 << "um^2" << endl;
	outfile << '\n' << endl;
	outfile.close();
}
