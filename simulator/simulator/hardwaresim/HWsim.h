// HWsim.h
#ifndef HWSIM_H_
#define HWSIM_H_

#include "typedef.h"
#include "InputParameter.h"
#include "Technology.h"
#include "MemCell.h"
#include "FunctionUnit.h"
#include "ReadParam.h"
#include "Core.h"
#include <vector>

class HWsim: public FunctionUnit {
public:
	HWsim();
	virtual ~HWsim() {}
	InputParameter inputParameter;
	Technology tech;
	MemCell cell;
	ReadParam params;

	/* Functions */
	void PrintProperty();
	void SaveOutput(int CoreIndex, const char* outputFile);
	void Initialize ();
	void CalculateArea();
	void CalculateLatency();
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
	int wireWidth;	/* The Width of the wire, unit: nm*/
	int numof1, numof2, numof3, numof4, numof5, numof6, numof7,
		numof8, numof9, numof10;

	Core* core;
};

#endif /* HWSIM_H_ */
