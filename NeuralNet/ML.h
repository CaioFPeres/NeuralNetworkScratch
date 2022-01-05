#pragma once
#include <iostream>
#include <stdlib.h>
#include <random>
#include <math.h>
#include <vector>
#include <string>
#include <locale>
#include <codecvt>
#include "Matrix.h"
#include <fstream>
#include <sstream>
#include <windows.h>

using std::cout;
using std::string;
using std::wstring;
using std::vector;


class ML
{

public:
	
	typedef struct Dimensions {
		int row;
		int col;
	}Dim;
	
	ML(Dim* dimensions);
	ML(string* files );
	~ML();

	void Train(string folderName);
	void Forward(Matrix* input);
	void Backward(Matrix* input, Matrix* label);
	void Activation(Matrix** hpre);
	
	Matrix* MLLoadImage(wstring fileName);

	bool CompareOutput(Matrix* m1, Matrix* m2);
	Matrix* LoadFile(string fileName, Dim dim);
	void SaveFile(string fileName, Matrix* m);
	void SaveAll();
	string eval(string fileName);
	vector<wstring>* getFilesNames(wstring directory, bool getFolders);
	void ReadZip(string fileName);


private:

	Matrix* w_i_h;
	Matrix* w_h_o;
	Matrix* b_i_h;
	Matrix* b_h_o;
	
	Matrix* delta_h;
	Matrix* delta_o;
	Matrix* h;
	Matrix* o;

	double learn_rate;
	double timesCorrect;
	int epochs;


};

