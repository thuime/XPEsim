// ActivationFunc.h
#ifndef ACTIVATIONFUNC_H_
#define ACTIVATIONFUNC_H_

#include "FunctionUnit.h"
#include "InputParameter.h"
#include "MemCell.h"
#include "ReadParam.h"
#include "Technology.h"
#include "typedef.h"

class ActivationFunc: public FunctionUnit
{
public:
	ActivationFunc(const InputParameter& _inputParameter,
			const Technology& _tech,
			const MemCell& _cell,
			const ReadParam& _readParam);
	virtual ~ActivationFunc() {}
	
	const InputParameter& inputParameter;
	const Technology& tech;
	const MemCell& cell;
	const ReadParam& readParam;
	
	/* Functions */
	void PrintProperty(const char* str);
	void SaveOutput(const char* str, const char* outputFile);
	void PrintMagicProperty();
	void Initialize(int _numCol, int _numIOBit, ActivationFunction _Func);
	void CalculateArea(double numInput, double IOBits,
			double _newHeight, double _newWidth,
		   	AreaModify _option);
	void CalculateLatency(double _rampInput, double _capLoad, double numRead);
	void CalculatePower(double numRead);

	/* Properties */
	bool initialized;
	int numCol;
	int numIOBit;
	ActivationFunction Func;

	double widthNandN, widthNandP;
	double widthInvN, widthInvP;

	double wNand, hNand;
	double wInv, hInv;

	double capNandInput, capNandOutput;
	double capInvInput, capInvOutput;
	
	double capLoad;
	double rampInput;

};

#endif
