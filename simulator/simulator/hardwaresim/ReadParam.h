#ifndef READPARAM_H_
#define READPARAM_H_

#include <string>
#include "typedef.h"

class ReadParam {
public:
	ReadParam();
	virtual ~ReadParam() {}

	/* Functions */
	void ReadParameterFromFile(const std::string & inputFile);
	void PrintParameter();

	/* Properties */
	DesignOptimization designopt;	/* Either read latency, write latency, read energy, write energy, leakage, or area */
	std::string designoptname;
	int processNode;				/* Process node (nm) */
	DeviceRoadmap deviceRoadmap;	/* ITRS roadmap: HP, LSTP */
	TransistorType transistorType;  /* Conventional CMOS, 2D FET, or TF    ET */
	std::string fileCellType;		/* Input file name of cell type: DigitalRRAMTHU */
	int temperature;				/* The ambient temperature, Unit: K */
	int numArrayCol;				/* The Number of array columns */
	int numArrayRow;				/* The Number of array rows */
	int WeightBits;					/* The Number of weight bits */
	int CellBits;					
	int IOBits;
	int Rmin;						/* Min resistance of cell, Unit: ohm */
	int Rmax;						/* Max resistance of cell, Unit: ohm */
	double ReadVoltage;				/* Read Voltage, Unit: V */
	double ResetVoltage;			/* Reset Voltage, Unit: V */
	double SetVoltage;				/* Set Voltage, Unit: V */
	double ResetPulseWidth;			/* Reset Pulse Width, Unit: s */
	double SetPulseWidth;			/* Set Pulse Width, Unit: s */
	double ReadPulseWidth;			/* Read Pulse Width, Unit: s */

	double accessVoltage;			/* Voltage on the gate of MOSFET in 1T1R cell, Unit: s */

	int multipleCells;				/* Using multiple cells as 1 cell */

	double clkFreq;					/* Clock frequence, Unit: Hz */
	int wireWidth;					/* Interwire Width, Unit: nm */
	int SABits;						/* SA bits */

	std::string outputFilePrefix;	/* Output file name */
};

#endif /* READPARAM_H_ */
