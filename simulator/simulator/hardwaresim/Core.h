// Core.h
#ifndef CORE_H_
#define CORE_H_

#include "typedef.h"
#include "InputParameter.h"
#include "Technology.h"
#include "MemCell.h"
#include "FunctionUnit.h"
#include "ReadParam.h"
#include "Sum.h"
#include "ShiftAdd.h"
#include "SubArray.h"

class Core: public FunctionUnit {
public:
	Core(InputParameter& _inputParameter, Technology& _tech, MemCell& _cell, ReadParam& _readParam);
	virtual ~Core() {}
	const InputParameter& inputParameter;
	const Technology& tech;
	const MemCell& cell;
	const ReadParam& readParam;

	/* Functions */
	void PrintProperty();
	void SaveOutput(const int& CoreIndex,const char* outputFile);
	void Initialize (double _unitWireRes);
	void CalculateArea();
	void CalculateLatency(double _rampInput);
	void CalculatePower(double activityRowRead);

	/* Properties */
	bool initialized;	/* Initialization flag */
	int numOutput;		/* Number of outputs */
	int numArrayCol;			/* Number of Array columns */
	int numArrayRow;
	int numCellPerSynapse;/* Number of cells per synapse */ 
	int numBitInput;
	int numWeightBit;
	int numCellBit;
	int numSABit;
	int numBitPartialSum;		/* The output bits of SA */
	double activityRowRead;
	int numof1, numof2, numof3, numof4, numof5, numof6, numof7,
		numof8, numof9, numof10;
	

	Sum sum_add;
	ShiftAdd shiftAdd;
	SubArray subArray;

};

#endif /* CORE_H_ */
