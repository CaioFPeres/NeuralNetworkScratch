#pragma once
#include <iostream>
#include <stdlib.h>
#include <random>
#include <math.h>
#include <vector>
#include <string>
#include <codecvt>

using std::string;
using std::wstring;
using std::cout;

class Matrix
{
public:

	double** matrix;
	int row;
	int col;


	Matrix(int row, int col, double low, double high);
	~Matrix();

	void Initialize(double low, double high);
	Matrix* Multiply(Matrix* m);
	Matrix* Sum(Matrix* m);
	Matrix* Subtract(Matrix* m);
	Matrix* ScalarDivision(double value);
	Matrix* ScalarMultiplication(double value);
	Matrix* ScalarMultiplication(Matrix* m);
	Matrix* Transpose();
	void Neg();
	void Exp();
	void Normalize(int value);
	void ArrayToMatrix(unsigned char* array);
	void OneHotEncode(int value, int maxValue);
	void Print();

private:
	void Free();
};

