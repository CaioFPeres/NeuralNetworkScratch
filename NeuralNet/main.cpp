#include "ML.h"


int main() {

	int size = 4;

	ML::Dim* dim = new ML::Dim[size];

	// neurons
	dim[0].row = 20;
	dim[0].col = 784;
	dim[1].row = 10;
	dim[1].col = 20;

	// biases
	dim[2].row = 20;
	dim[2].col = 1;
	dim[3].row = 10;
	dim[3].col = 1;
	
	// to load saved neural network
	/*
	string files[4];
	files[0] = ".\\savedNet\\w_i_h.txt";
	files[1] = ".\\savedNet\\w_h_o.txt";
	files[2] = ".\\savedNet\\b_i_h.txt";
	files[3] = ".\\savedNet\\b_h_o.txt";
	*/

	// to load saved neural network
	// ML* mainML = new ML(files);

	ML* mainML = new ML(dim);


	mainML->Train("C:\\Users\\Caio\\source\\repos\\NeuralNet\\NeuralNet\\images\\");
	mainML->SaveAll();

	delete mainML;

	system("pause");

	return 0;

}