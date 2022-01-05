#include "ML.h"

#define STB_IMAGE_IMPLEMENTATION
#include "./stb-master/stb_image.h"

ML::ML(Dim* dimensions) {

	learn_rate = 0.01;
	timesCorrect = 0;
	epochs = 3;

	// need to initialize structs

	this->delta_h = nullptr;
	this->delta_o = nullptr;
	this->h = nullptr;
	this->o = nullptr;
	
	this->w_i_h = new Matrix(dimensions[0].row, dimensions[0].col, -0.5, 0.5);
	this->w_h_o = new Matrix(dimensions[1].row, dimensions[1].col, -0.5, 0.5);

	this->b_i_h= new Matrix(dimensions[2].row, dimensions[2].col, -0.5, 0.5);
	this->b_h_o = new Matrix(dimensions[3].row, dimensions[3].col, -0.5, 0.5);
}

ML::ML(string* files) {
	learn_rate = 0.01;
	timesCorrect = 0;
	epochs = 1;

	Dim dim1;
	dim1.row = 20;
	dim1.col = 784;
	Dim dim2;
	dim2.row = 10;
	dim2.col = 20;

	Dim dim3;
	dim3.row = 20;
	dim3.col = 1;
	Dim dim4;
	dim4.row = 10;
	dim4.col = 1;

	
	this->w_i_h = LoadFile(files[0], dim1);
	this->w_h_o = LoadFile(files[1], dim2);
	this->b_i_h = LoadFile(files[2], dim3);
	this->b_h_o = LoadFile(files[3], dim4);
	
}

ML::~ML() {
	delete w_i_h;
	delete w_h_o;
	delete b_i_h;
	delete b_h_o;

	delete delta_h;
	delete delta_o;
	delete h;
	delete o;
}

string ML::eval(string fileName) {

	using convert_type = std::codecvt_utf8<wchar_t>;
	std::wstring_convert<convert_type, wchar_t> converter;
	std::wstring wfileName = converter.from_bytes(fileName);

	Matrix* img = MLLoadImage(wfileName);

	Forward(img);

	double highest = 0;
	int highestIndex = 0;
	
	for (int i = 0; i < this->o->row; i++){
		if (o->matrix[i][0] > highest) {
			highest = o->matrix[i][0];
			highestIndex = i;
		}
	}

	cout << highest << '\n';
	
	delete img;
	return string(std::to_string(highestIndex));
}

void ML::SaveAll() {
	SaveFile(".\\savedNet\\w_i_h.txt", w_i_h);
	SaveFile(".\\savedNet\\w_h_o.txt", w_h_o);
	SaveFile(".\\savedNet\\b_i_h.txt", b_i_h);
	SaveFile(".\\savedNet\\b_h_o.txt", b_h_o);
}

void ML::SaveFile(string fileName, Matrix* m) {
	
	std::ofstream file;
	file.open(fileName, std::ios::out);

	file.precision(17);

	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) {
			file << m->matrix[i][j] << ' ';
		}
	}

	file.close();
}

Matrix* ML::LoadFile(string fileName, Dim dim) {

	std::ifstream file;
	string buffer;
	vector<double> values;

	int k = 0;

	file.open(fileName);
	file.precision(17);

	
	while (file.good()) {
		file >> buffer;
		
		std::istringstream SS(buffer);
		double d;
		SS >> d;
		values.push_back(d);
	}

	
	Matrix* result = new Matrix(dim.row, dim.col, 0, 0);

	for (int i = 0; i < result->row; i++)
		for (int j = 0; j < result->col; j++)
			result->matrix[i][j] = values[k++];


	file.close();
	return result;
}

void ML::Forward(Matrix* input) {

	if (h) {
		delete h;
		h = nullptr;
	}
	if (o) {
		delete o;
		o = nullptr;
	}
	
	// input -> hidden
	Matrix* mult = w_i_h->Multiply(input);
	h = b_i_h->Sum(mult);

	delete mult;
	Activation(&h);


	// hidden -> output
	mult = w_h_o->Multiply(h);
	o = b_h_o->Sum(mult);
	
	delete mult;
	Activation(&o);

}

void ML::Backward(Matrix* input, Matrix* label) {

	if (delta_h) {
		delete delta_h;
		delta_h = nullptr;
	}
	if (delta_o) {
		delete delta_o;
		delta_o = nullptr;
	}

	// output -> hidden (cost function derivative)
	delta_o = o->Subtract(label);


	// w_h_o
	Matrix* hT = h->Transpose();
	Matrix* mult = delta_o->Multiply(hT);

	Matrix* multLearnRate = mult->ScalarMultiplication(-learn_rate);
	Matrix* newW_H_O = w_h_o->Sum(multLearnRate);
	
	if (w_h_o){
		delete w_h_o;
		w_h_o = nullptr;
	}

	w_h_o = newW_H_O;

	delete hT;
	delete mult;
	delete multLearnRate;


	// b_h_o
	multLearnRate = delta_o->ScalarMultiplication(-learn_rate);
	Matrix* newB_H_O = b_h_o->Sum(multLearnRate);

	if (b_h_o) {
		delete b_h_o;
		b_h_o = nullptr;
	}

	b_h_o = newB_H_O;

	delete multLearnRate;


	// hidden -> input (activation function derivative)
	Matrix* Ones = new Matrix(h->row, h->col, 1 ,1);

	Matrix* newH = Ones->Subtract(h);
	Matrix* multH = h->ScalarMultiplication(newH);
	
	Matrix* w_h_o_T = w_h_o->Transpose();
	Matrix* multT = w_h_o_T->Multiply(delta_o);
	delta_h = multT->ScalarMultiplication(multH);


	Matrix* imgT = input->Transpose();
	Matrix* multDelta = delta_h->Multiply(imgT);
	multLearnRate = multDelta->ScalarMultiplication(-learn_rate);
	Matrix* newW_I_H = w_i_h->Sum(multLearnRate);
	
	if (w_i_h) {
		delete w_i_h;
		w_i_h = nullptr;
	}

	w_i_h = newW_I_H;

	delete multLearnRate;

	multLearnRate = delta_h->ScalarMultiplication(-learn_rate);
	Matrix* newB_I_H = b_i_h->Sum(multLearnRate);

	if (b_i_h){
		delete b_i_h;
		b_i_h = nullptr;
	}

	b_i_h = newB_I_H;

	delete Ones;
	delete newH;
	delete multH;
	delete w_h_o_T;
	delete multT;
	delete imgT;
	delete multDelta;
	delete multLearnRate;
	
}

//sigmoid function
void ML::Activation(Matrix** pre) {
	
	(*pre)->Neg();
	(*pre)->Exp();

	Matrix* Ones = new Matrix((*pre)->row, (*pre)->col, 1, 1);

	Matrix* sum = Ones->Sum(*pre);
	
	delete (*pre);

	*pre = sum->ScalarDivision(1);

	delete Ones;
	delete sum;
}

void ML::Train(string folderName) {

	using convert_type = std::codecvt_utf8<wchar_t>;
	std::wstring_convert<convert_type, wchar_t> converter;
	std::wstring wfolderName = converter.from_bytes(folderName);
	std::string currentFolder;

	int totalNumOfFiles = 0;
	
	// getting files from specified folder
	vector<wstring>* folders = getFilesNames(wfolderName, true);
	vector<vector<wstring>*>* files = new vector<vector<wstring>*>();

	// each index is a folder, and we're pushing every folder's files in that index
	
	for (int i = 2; i < folders->size(); i++) {
		files->push_back(getFilesNames(wfolderName + (*folders)[i], false));
		totalNumOfFiles = totalNumOfFiles + (*files)[i-2]->size();
	}

	

	Matrix** label = new Matrix*[files->size()]();

	for (int i = 0; i < files->size(); i++) {
		label[i] = new Matrix(files->size(), 1, 0, 0);
		label[i]->OneHotEncode(i, files->size());
	}

	int numOfImages = 0;

	int* j = new int[files->size()];
	for (int i = 0; i < files->size(); i++) {
		j[i] = 0;
	}
	
	
	for (int epo = 0; epo < epochs; epo++) {

		while (numOfImages != totalNumOfFiles) {
			
			for (int i = 0; i < files->size(); i++) {

				if (j[i] < (*files)[i]->size()) {

					Matrix* img = MLLoadImage(wfolderName + wstring(std::to_wstring(i)) + wstring(L"\\") + wstring((*(*files)[i])[j[i]]));
					
					// training loop goes here
					
					Forward(img);

					timesCorrect += (int)CompareOutput(o, label[i]);

					Backward(img, label[i]);
					
					delete img;
					
					numOfImages++;
					j[i]++;

				}
			}
		
		}

		//print accuracy:
		cout << (timesCorrect / numOfImages) * 100 << '%' << '\n';
		cout << timesCorrect << '\n';
		
		//reset variables
		timesCorrect = 0;
		numOfImages = 0;
		for (int i = 0; i < files->size(); i++) {
			j[i] = 0;
		}

	}
	
	folders->clear();
	delete folders;
	
	
	for (int i = 0; i < files->size(); i++) {
		(*files)[i]->clear();
		delete (*files)[i];
	}

	
	files->clear();
	delete files;
	delete[] label;
	delete[] j;

}

vector<wstring>* ML::getFilesNames(wstring directory, bool getFolders) {

	vector<wstring>* fileNames = new vector<wstring>;
	LPTSTR lpDir = (LPTSTR)directory.c_str();
	int dirNameSize = 0;

	WIN32_FIND_DATA ffd;
	LARGE_INTEGER filesize;
	TCHAR szDir[MAX_PATH];
	HANDLE hFind = INVALID_HANDLE_VALUE;
	DWORD dwError = 0;

	// Prepare string for use with FindFile functions. First, copy the
	// string to a buffer, then append '\*' to the directory name.

	wcscpy_s(szDir, lpDir);

	for (; szDir[dirNameSize] != '\0'; dirNameSize++);

	szDir[dirNameSize++] = '\\';
	szDir[dirNameSize++] = '*';
	szDir[dirNameSize] = '\0';


	// Find the first file in the directory.

	hFind = FindFirstFile(szDir, &ffd);

	if (INVALID_HANDLE_VALUE == hFind)
	{
		wprintf_s(L"\nERROR: %lu", dwError);
		return fileNames;
	}

	// List all the files in the directory with some info about them.
	do
	{
		if (ffd.dwFileAttributes == FILE_ATTRIBUTE_DIRECTORY)
		{
			// found directory
			if(getFolders)
				fileNames->push_back(ffd.cFileName);
		}
		else
		{
			//filesize.LowPart = ffd.nFileSizeLow;
			//filesize.HighPart = ffd.nFileSizeHigh;
			//wprintf_s(TEXT("  %s   %ld bytes\n"), ffd.cFileName, (int)filesize.QuadPart);  // printa nomes e tamanhos dos arquivos;
			if (!getFolders)
				fileNames->push_back(ffd.cFileName);
		}


	} while (FindNextFile(hFind, &ffd) != 0);

	dwError = GetLastError();
	if (dwError != ERROR_NO_MORE_FILES)
	{
		wprintf_s(L"\nERROR: %lu", dwError);
	}

	// printing with C++ iterator
	/*
	for (std::vector<wstring>::iterator it = (*fileNames).begin(); it != (*fileNames).end(); ++it) {
		wcout << *it << endl;
	}
	*/

	FindClose(hFind);

	return fileNames;
}

bool ML::CompareOutput(Matrix* m1, Matrix* m2) {

	if (m1->row != m2->row || m1->col != m2->col) {
		cout << "\nCan't Compare, different dimensions!\n";
		return NULL;
	}

	double highestValue1 = m1->matrix[0][0];
	int highestIndexRow1 = 0;
	int highestIndexCol1 = 0;

	double highestValue2 = m2->matrix[0][0];
	int highestIndexRow2 = 0;
	int highestIndexCol2 = 0;


	for (int i = 0; i < m1->row; i++) {
		for (int j = 0; j < m1->col; j++) {
			
			if (m1->matrix[i][j] > highestValue1) {
				highestValue1 = m1->matrix[i][j];
				highestIndexRow1 = i;
				highestIndexCol1 = j;
			}
			if (m2->matrix[i][j] > highestValue2) {
				highestValue2 = m2->matrix[i][j];
				highestIndexRow2 = i;
				highestIndexCol2 = j;
			}
				
		}
	}

	if (highestIndexRow1 == highestIndexRow2)
		return true;
	else
		return false;
}

// returns normalized matrix
Matrix* ML::MLLoadImage(wstring fileName) {

	//convert wstring to string
	using convert_type = std::codecvt_utf8<wchar_t>;
	std::wstring_convert<convert_type, wchar_t> converter;

	std::string stringFileName = converter.to_bytes(fileName);

	int width, height, bpp;

	unsigned char* data = stbi_load(stringFileName.c_str(), &width, &height, &bpp, 0);

	Matrix* img = new Matrix(height*width, 1, 0, 0);
	img->ArrayToMatrix(data);
	img->Normalize(255);

	return img;
}

void ML::ReadZip(string fileName) {


}
