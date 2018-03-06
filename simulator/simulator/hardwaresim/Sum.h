// Sum.h
#ifndef SUM_H_
#define SUM_H_

#include "typedef.h"
#include "InputParameter.h"
#include "Technology.h"
#include "MemCell.h"
#include "FunctionUnit.h"
#include "Adder.h"

class Sum: public FunctionUnit {
public:
	Sum(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell);
	virtual ~Sum() {}
	const InputParameter& inputParameter;
	const Technology& tech;
	const MemCell& cell;

	/* Functions */
	void PrintProperty(const char* str);
	void SaveOutput(const char* str, const char* outputFile);
	void PrintMagicProperty();
	void Initialize(int _numCol, int _numCellPerSynapse, int _numSumBit);
	void CalculateArea(double heightArray, double widthWidth, AreaModify _option);
	void CalculateLatency(double numRead);
	void CalculatePower(double numRead);

	/* Properties */
	bool initialized;	/* Initialization flag */
	int numOutput;		/* Number of outputs */
	int numCol;			/* Number of Array columns */
	int numSumPerCol;	
	int numSumBit;		/* Number of bits of S/A output */
	int numAdder;		/* Number of Adders in sum circuit */
	int numAdderDepth;

	Adder adder;
};

#endif /* SUM_H_ */
