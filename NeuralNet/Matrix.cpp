#include "Matrix.h"

/*
#define STB_IMAGE_IMPLEMENTATION
#include "./stb-master/stb_image.h"
*/

Matrix::Matrix(int row, int col, double low, double high) {

	this->row = row;
	this->col = col;

	Initialize(low, high);
}

Matrix::~Matrix() {
	Free();
}

void Matrix::Initialize(double low, double high) {

	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(low, high);

	if (this->row == NULL || this->col == NULL) {
		cout << "Num of rows or columns not defined!\n";
		return;
	}

	this->matrix = new double* [this->row];

	for (int i = 0; i < this->row; i++) {
		this->matrix[i] = new double[this->col];
	}

	//(* matrix)[i][j] = (((double)rand() / RAND_MAX) * (high - low)) + low;

	for (int i = 0; i < this->row; i++) {
		for (int j = 0; j < this->col; j++) {
			this->matrix[i][j] = distribution(generator);
		}
	}
}

Matrix* Matrix::Multiply(Matrix* m) {

	if (this->col != m->row) {
		cout << "\nCan't multiply, different dimensions! m1 col: " << this->col << " " << "m2 row: " << m->row;
		return NULL;
	}

	Matrix* result = new Matrix(this->row, m->col, 0, 0);


	for (int i = 0; i < this->row; i++) {
		for (int j = 0; j < m->col; j++) {
			result->matrix[i][j] = 0;

			for (int k = 0; k < m->row; k++) {
				result->matrix[i][j] += this->matrix[i][k] * m->matrix[k][j];
			}
		}
	}

	return result;
}

Matrix* Matrix::Sum(Matrix* m) {

	if (this->row != m->row || this->col != m->col) {
		cout << "\nCan't add, different dimensions!\n";
		return NULL;
	}

	Matrix* result = new Matrix(this->row, this->col, 0, 0);

	for (int i = 0; i < this->row; i++)
		for (int j = 0; j < this->col; j++)
			result->matrix[i][j] = this->matrix[i][j] + m->matrix[i][j];

	return result;
}

Matrix* Matrix::Subtract(Matrix* m) {

	if (this->row != m->row || this->col != m->col) {
		cout << "\nCan't subtract, different dimensions!\n";
		return NULL;
	}

	Matrix* result = new Matrix(this->row, this->col, 0, 0);

	for (int i = 0; i < this->row; i++)
		for (int j = 0; j < this->col; j++)
			result->matrix[i][j] = this->matrix[i][j] - m->matrix[i][j];

	return result;
}

// value / each element
Matrix* Matrix::ScalarDivision(double value) {

	Matrix* result = new Matrix(this->row, this->col, value, value);

	for (int i = 0; i < this->row; i++)
		for (int j = 0; j < this->col; j++)
			result->matrix[i][j] = result->matrix[i][j] / this->matrix[i][j];

	return result;
}

Matrix* Matrix::ScalarMultiplication(double value) {

	Matrix* result = new Matrix(this->row, this->col, value, value);

	for (int i = 0; i < this->row; i++)
		for (int j = 0; j < this->col; j++)
			result->matrix[i][j] = result->matrix[i][j] * this->matrix[i][j];

	return result;
}

Matrix* Matrix::ScalarMultiplication(Matrix* m) {

	if (this->row != m->row || this->col != m->col) {
		cout << "\nCan't multiply, different dimensions!\n";
		return NULL;
	}

	Matrix* result = new Matrix(this->row, this->col, 0, 0);

	for (int i = 0; i < this->row; i++)
		for (int j = 0; j < this->col; j++)
			result->matrix[i][j] = this->matrix[i][j] * m->matrix[i][j];

	return result;
}

Matrix* Matrix::Transpose() {

	Matrix* result = new Matrix(this->col, this->row, 0, 0);

	for (int i = 0; i < result->row; i++)
		for (int j = 0; j < result->col; j++) {
			result->matrix[i][j] = this->matrix[j][i];
		}

	return result;
}

void Matrix::Neg() {
	for (int i = 0; i < this->row; i++) {
		for (int j = 0; j < this->col; j++) {
			if (this->matrix[i][j] != 0)
				this->matrix[i][j] = -this->matrix[i][j];
		}
	}
}

void Matrix::Exp() {

	double e = 2.7182818284590452353602874713527;

	for (int i = 0; i < this->row; i++)
		for (int j = 0; j < this->col; j++)
			this->matrix[i][j] = pow(e, (this->matrix[i][j]));
}

//frees the whole structure
void Matrix::Free() {

	for (int i = 0; i < this->row; i++)
	{
		delete[] this->matrix[i];
		this->matrix[i] = nullptr;
	}

	delete[] this->matrix;
	this->matrix = nullptr;
}

void Matrix::Normalize(int value) {
	for (int i = 0; i < this->row; i++) {
		for (int j = 0; j < this->col; j++) {
			this->matrix[i][j] = this->matrix[i][j] / value;
		}
	}
}

// frees array automatically
void Matrix::ArrayToMatrix(unsigned char* array) {

	for (int i = 0; i < this->row; i++) {
		this->matrix[i][0] = array[i];
	}

	delete[] array;
}

void Matrix::OneHotEncode(int value, int maxValue) {

	for (int i = 0; i < maxValue; i++) {
		if (i == value)
			this->matrix[i][0] = 1;
		else
			this->matrix[i][0] = 0;
	}

}
/*
// returns normalized matrix
Matrix* Matrix::MLLoadImage(wstring fileName) {

	//convert wstring to string
	using convert_type = std::codecvt_utf8<wchar_t>;
	std::wstring_convert<convert_type, wchar_t> converter;

	std::string stringFileName = converter.to_bytes(fileName);

	int width, height, bpp;

	unsigned char* data = stbi_load(stringFileName.c_str(), &width, &height, &bpp, 0);

	Matrix* img = ArrayToMatrix(data, height, width);

	img->NormalizeMatrix(255);

	return img;
}
*/

void Matrix::Print() {

	for (int i = 0; i < this->row; i++) {
		for (int j = 0; j < this->col; j++) {
			cout << this->matrix[i][j] << ' ';
		}
		cout << '\n';
	}
}