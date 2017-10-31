// Parameterised Neural Net With Sigmoid.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#define TXTFILE ".txt"
#define NETWORKNUMBER ""
#define FILEPATH "C:\\Users\\c3164126\\Documents\\MyStuff\\trunk\\PhD\\My Papers\\Grid Optimisation using Neural Networks\\"
#define TRAINING 1
#define LEARNING_RATE 0.3
double sigmoid(double x);
int max_value(int* x, int num_values);
void randomise(double* input, int inputcomponents);
void set_to_1(double* input, int inputcomponents);
void set_to_0(double* input, int inputcomponents);
void net_layer(double* input, int num_of_inputs, double* output, int num_of_outputs, double* w, double* b, double* dpred_dout, int training_flag);

void training(double*dpred_dout,double* dout_dw, double* dprev, int num_neurons_n, int num_neurons_1, int num_neurons_0, double* w, double* b);

//defining globals
int training_flag = 0;



int main()
{

	
	//////////////////////////////////////////////////////////// READING FROM INITIALISATION FILE //////////////////////////////////////////////////////////////////////////////////////////

	//Opening the initialisation file
	FILE *init_file;
	init_file = fopen(FILEPATH"NN_init" NETWORKNUMBER TXTFILE , "r");
	if (init_file == NULL)
	{
		printf("ERROR! can't open initialisation file to read, check file path!\n");
		return 0;
	}
	else printf("Initialisation file successfully opened\n");
		
	int num_layers, data_volume;
	double iterations;
	fscanf(init_file, "%d\t %lf\t %d\t %d\t", &training_flag, &iterations, &num_layers, &data_volume);
	//Error checking for the initialisation file starts here
	if ((training_flag < 0) | (training_flag > 2))
	{
		printf("ERROR! Training flag must be the values 0, 1 or 2! Check initialisation file!\nExiting program now\n");
		return(1);
	}
	if ((iterations < 0))
	{
		printf("ERROR! Iterations must be greater than 0! Check initialisation file!\nExiting program now\n");
		return(1);
	}
	if ((num_layers < 2))
	{
		printf("ERROR! Number of layers must be greater than 2! Check initialisation file!\nExiting program now\n");
		return(1);
	}
	if ((data_volume < 1))
	{
		printf("ERROR! Data entries must be > 0! Check initialisation file!\nExiting program now\n");
		return(1);
	}
	//Setting up how many neurons are in each layer of the neural network by the values in the file.
	int* num_neurons = (int*)malloc(sizeof(int) * (num_layers));
	int e = 0;
	while (fscanf(init_file, "%d", &num_neurons[e]) == 1)
	{
		if((num_neurons[e]<1))
		{
			printf("ERROR! Layers must have at least one neuron! Check initialisation file!\nExiting program now\n");
			return(1);
		}
		e++;
	}
	//////////////////////////////////////////////////////////// FINISHED READING FROM INITIALISATION FILE //////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////// READING FROM DATA FILE //////////////////////////////////////////////////////////////////////////////////////////
	//Opening the Data file
	FILE *data_file;
	data_file = fopen(FILEPATH"NN_data" NETWORKNUMBER TXTFILE, "r");
	if (init_file == NULL)
	{
		printf("ERROR! can't open data file to read, check file path!\n");
		return 0;
	}
	else printf("Data file successfully opened\n");

	double* target	= (double*)malloc(sizeof(double) * num_neurons[num_layers - 1] * data_volume);		//initialising the target array (matrix) size
	double* data	= (double*)malloc(sizeof(double) * data_volume * num_neurons[0]);				//initialising the data array (matrix) size
	if (e != (num_layers)) 
	{
		printf("ERROR! Data in initialisation file does not have the correct number of entries!\nCheck the entries in the initialisation file is correct!");
		return(0);
	}
	else printf("Initialisation completed successfully\n");
	fclose(init_file);


	e = 0; //resetting error flag
	
	//taking data from file and storing it in relevant information
	while (!feof(data_file)) 
	{
		//This loop grabs the data for the specific entry and stores it in the data array.
		for (int i = 0; i < num_neurons[0]; i++) {
			fscanf(data_file, "%lf", &data[num_neurons[0] * e + i]);
			printf("%lf\t", data[num_neurons[0] * e + i]);
		}
		//If the neural network is getting trained, store the target variables
		if ((training_flag == 1)|(training_flag == 2))
		{
			for (int i = 0; i < num_neurons[num_layers - 1]; i++)	
			{
				fscanf(data_file, "%lf", &target[num_neurons[(num_layers - 1)] * e + i]);
				printf("%lf\t", target[num_neurons[(num_layers - 1)] * e + i]);
			}
			
		}
		printf("e:%d", e);
			printf("\n");
		e++;
	}
	if (e != (data_volume))
	{
		printf("ERROR! Data in data file does not have the correct number of entries!\nCheck the entires in the data file and initialisation file is correct!");
		return(0);
	}
	else printf("Data reading completed successfully\n");
	fclose(data_file);
	//////////////////////////////////////////////////////////// FINISHED READING FROM DATA FILE //////////////////////////////////////////////////////////////////////////////////////////
	
	
	//////////////////////////////////////////////////////////// Initialising variables with specified size by files///////////////////////////////////////////////////////////////////////
	double**	layer =			(double**)malloc(sizeof(double*)	* num_layers);
	double**	w =				(double**)malloc(sizeof(double*)	* (num_layers - 1));
	double**	b =				(double**)malloc(sizeof(double*)	* (num_layers - 1));
	double**	dpred_dout =	(double**)malloc(sizeof(double*)	* (num_layers - 1));


	//layer[0] is being initialised here so that everything else can be in one for loop.
	layer[0] = (double*)malloc(sizeof(double) * num_neurons[0]);

	for (int i = 0; i < (num_layers - 1); i++)
	{
		layer[i + 1] = (double*)calloc(num_neurons[i + 1], sizeof(double));								
		dpred_dout[i] = (double*)calloc(num_neurons[i + 1], sizeof(double));							
		w[i] = (double*)calloc(num_neurons[i] * num_neurons[i + 1], sizeof(double));					
		b[i] = (double*)calloc(num_neurons[i + 1], sizeof(double));								
																	
		//if the system is being trained for the first time. 
		if (training_flag == 1)
		{
			time_t t;
			srand(time(&t));
			randomise(&w[i][0], num_neurons[i] * num_neurons[i + 1]);
			randomise(&b[i][0], num_neurons[i + 1]);
		}
	}
	//if the system is assuming an already trained neural net, open and load the weights and biasses between each layer.
	if ((training_flag==0)|(training_flag == 2))
	{
		FILE* w_file;
		w_file = fopen(FILEPATH"NN_w" NETWORKNUMBER TXTFILE, "r");
		if (w_file == NULL)
		{
			printf("ERROR! can't open weight file to read, check file path!\n");
			return 0;
		}
		else printf("Weight file successfully opened\n");

		FILE* b_file;
		b_file = fopen(FILEPATH"NN_b" NETWORKNUMBER TXTFILE, "r");
		if (b_file == NULL)
		{
			printf("ERROR! can't open bias file to read, check file path!\n");
			return 0;
		}
		else printf("Bias file successfully opened\n");
		e = 0;
		while( !feof(w_file) )
		{
			for (int j = 0; j < num_neurons[e]*num_neurons[e+1]; j++)
			{
				fscanf(w_file, "%lf\t", &w[e][j]);
					
			}
			
			e++;
			if (e == (num_layers - 1) && !feof(w_file))
			{
				break;
			}
		}
		if ((e != (num_layers - 1)))
		{
			printf("Error! Weight file has incorrect entries! Please re-train system or fix issue\nProgram has not exited\n");
			return 0;
		}
		else printf("Weight values successfully read from file\n");

		e = 0;
		while(!feof(b_file))
		{
			for (int j = 0; j < num_neurons[e + 1]; j++)
			{
				if (!feof(b_file)) fscanf(b_file, "%lf\t", &b[e][j]);
			}
			e++;
		}
		if ((e != (num_layers - 1)))
		{
			printf("Error! Bias file has incorrect entries! Please re-train system or fix issue\nProgram has not exited\n");
			return 0;
		}
		else printf("Bias values successfully read from file\n");
		e = 0;
		fclose(w_file);
		fclose(b_file);

	}
	/////////////////////////////////////////////////////////////		EVALUATION STATE STARTS HERE		//////////////////////////////////////////////////////////
	//In evaluation state (training_flag = 0), the system does not perform back-propagation and is only used to take inputs and make predictions on what the output should
	//be from those inputs. Input data alone with the estimates are stored in NN_eval
	if (training_flag == 0)
	{
		//opening the file the data with result will be stored in.
		FILE* eval_file;
		eval_file = fopen(FILEPATH"NN_eval" NETWORKNUMBER TXTFILE, "w");
		if (eval_file == NULL)
		{
			printf("ERROR! Failed to create result information file, check file path!\n");
			return 0;
		}
		else printf("Result file successfully opened\n");

		//data evaluation loop. In this loop the trained NN will be receive data, make a prediction and then save the data set plus the results into a file.
		for (int i = 0; i < data_volume; i++)
		{
			for (int j = 0; j < num_neurons[0]; j++)
			{
				layer[0][j] = data[i*num_neurons[0] + j]; //updating the input buffer/layer
				fprintf(eval_file, "%lf\t", data[i*num_neurons[0] + j]);
			}
			/////////////////////////////////////////////////////////////		Forward propagation					/////////////////////////////////////////////////////////////
			for (int j = 0; j < (num_layers - 1); j++)
			{
				net_layer(&layer[j][0], num_neurons[j], &layer[j + 1][0], num_neurons[j + 1], &w[j][0], &b[j][0], &dpred_dout[j][0],  0);
			}
			for(int j = 0; j<num_neurons[num_layers - 1]; j++) fprintf(eval_file, "%lf\t", layer[num_layers - 1][j]);
			if(i != (data_volume-1)) fprintf(eval_file, "\n");
		}
		
		
		for (int i = 0; i < num_neurons[0]; i++) 
		{
			fscanf(data_file, "%lf", &data[num_neurons[0] * e + i]);
		}
		for (int i = 0; i < num_neurons[num_layers - 1]; i++)
		{
			fscanf(data_file, "%lf", &target[num_neurons[(num_layers - 1)] * e + i]);
		}
		fclose(eval_file);
	}
	/////////////////////////////////////////////////////////////		EVALUATION STATE FINISHES HERE		//////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////		TRAINING STATE STARTS HERE			//////////////////////////////////////////////////////////
	//In training state the data is propagated forwards, where the result is then compared to target values creating a cost squared function. The partial derivative 
	//of the cost function is then propagated backwards through the system, adjusting the weight bias values of each layer to improve the performance of the system.
	//Back propagation uses calculus and linear algebra to achieve this, the matrix size is optimised to improve the speed of the system by removing unecessary rows
	//and columns. ASK CAMERON FOR EXPLANATION IF YOU GET LOST
	else
	{
		int x = 0;
		//creating temporary square matrix which will hold intermediate values of the cascading partial derivative between layers
		double* dprev = (double*)calloc(num_neurons[num_layers - 1], sizeof(double));			
		for (int x = 0; x < iterations; x++)
		{
			int d = rand() % (int)data_volume;	//choosing a random data point to train the network
			#pragma omp parallel for
			for (int j = 0; j < num_neurons[0]; j++) layer[0][j] = data[d*num_neurons[0] + j]; //updating the input buffer/layer

			/////////////////////////////////////////////////////////////		Forward propagation					/////////////////////////////////////////////////////////////
			for (int j = 0; j < (num_layers - 1); j++)
			{
				net_layer(&layer[j][0], num_neurons[j], &layer[j + 1][0], num_neurons[j + 1], &w[j][0], &b[j][0], &dpred_dout[j][0],  1);
			}

			/////////////////////////////////////////////////////////////		FORWARD PROPAGATION FINISHES HERE!		//////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////		BACK PROPAGATION STARTS HERE!		/////////////////////////////////////////////////////////////
			#pragma omp parallel for
			for (int k = 0; k < num_neurons[num_layers - 1]; k++)
			{
				dprev[k] = 2 * (layer[num_layers - 1][k] - target[d*num_neurons[num_layers - 1] + k]);
			}
			for (int j = (num_layers - 1); j > 0; j--)
			{
				training(&dpred_dout[j-1][0], &layer[j-1][0], &dprev[0], num_neurons[num_layers - 1], num_neurons[j], num_neurons[j - 1], &w[j - 1][0], &b[j - 1][0]);
			}
			/////////////////////////////////////////////////////////////		FORWARD PROPAGATION FINISHES HERE!		//////////////////////////////////////////////////////////
		}
		free(dprev);
		FILE *w_file;
		FILE *b_file;
		w_file = fopen(FILEPATH"NN_w" NETWORKNUMBER TXTFILE, "w");
		if (w_file == NULL)
		{
			printf("ERROR! Failed to create weight information file, check file path!\n");
			return 0;
		}
		else printf("Weight file successfully opened\n");
		b_file = fopen(FILEPATH"NN_b" NETWORKNUMBER TXTFILE, "w");
		if (w_file == NULL)
		{
			printf("ERROR! Failed to create bias information file, check file path!\n");
			return 0;
		}
		else printf("Bias file successfully opened\n");
		for (int i = 0; i < (num_layers - 1); i++)
		{
			for (int j = 0; j < (num_neurons[i] * num_neurons[i + 1]); j++)
			{
				fprintf(w_file, "%lf\t", w[i][j]);
			}
			for (int j = 0; j < num_neurons[i + 1]; j++) fprintf(b_file, "%lf\t", b[i][j]);
			if (i < (num_layers - 2))
			{
				fprintf(w_file, "\n");
				fprintf(b_file, "\n");
			}
		}
		fclose(w_file);
		fclose(b_file);
		printf("Weights and Biasses saved!\n");
	}
	

	//free(layer[0]);
	for (int i = 0; i < (num_layers-1); i++)
	{
		free(layer[i]);
		free(w[i]);
		free(b[i]);
		free(dpred_dout[i]);
	}
	free(layer[num_layers - 1]);
	free(layer);
	free(w);
	free(b);
	free(dpred_dout);
	free(num_neurons);
	return 0;
}

//This function computes the maximum value of an int array.
int max_value(int* x, int num_values)
{
	int max_value = 0;
	for (int i = 0; i < num_values; i++)
	{
		if (x[i] > max_value) max_value = x[i];
	}
	return(max_value);
}

//This function is the used as the activation function of the neural network. This bit can be removed from the code and swapped out with other activation functions.
double sigmoid(double x)
{
	return(1 / (1 + exp(-x)));
}

//this function is used to randomise the weights between -1.0 and 1.0 upon a fresh initialisation of a neural network. This can be omitted if trained weight and bias values are already known.
void randomise(double* input, int inputcomponents)
{
	//printf("the size of array is: %d\n", sizeof(input[0]));
	#pragma omp parallel for
	for (int i = 0; i < inputcomponents; i++)
	{
		input[i] = (double)rand() / RAND_MAX*2.0 - 1.0;
	}
	return;
}

//sets the values of a double array to 1.0, this is used to initialise the partial differential temporary value holder
void set_to_1(double* input, int inputcomponents)
{
	//printf("the size of array is: %d\n", sizeof(input[0]));
	#pragma omp parallel for
	for (int i = 0; i < inputcomponents; i++)
	{
		input[i] = 1;
	}
	return;
}

//sets the values of a double array to 0.0, this is used to initialise the partial differential temporary value holder
void set_to_0(double* input, int inputcomponents)
{
	//printf("the size of array is: %d\n", sizeof(input[0]));
	#pragma omp parallel for
	for (int i = 0; i < inputcomponents; i++)
	{
		input[i] = 0;
	}
	return;
}

//this function is used to update the neuron values for forward propagation, if the mode is training (training_flag=1) the intermediate partial derivatives are also calculated
//so that they can be consolidated for use in the training function of the neural network.
void net_layer(double* input, int num_of_inputs, double* output, int num_of_outputs, double* w, double* b, double* dpred_dout, int training_flag)
{
	//Updating the output buffer and performing training for this layer
	//This will compute each output one by one and then squash the result into a sigmoid function. 
	//Within this loop the differential terms are also created, this is done in this function because the intermediate term dpred_dout is only able to be computed in this process.
	#pragma omp parallel for
	for (int j = 0; j < (num_of_outputs); j++)
	{
		output[j] = 0; //clearing the output so that recursive summing can be carried out
		for (int i = 0; i < num_of_inputs; i++)
		{
			output[j] += input[i] * w[i*num_of_outputs + j];
		}
		output[j] += b[j];
		double temp = sigmoid(output[j]);
		//Checking if the network is being trained, if so it will populate the two different partial derivative matrices corresponding to the current input
		if (training_flag == 1)
		{
			//NOTE dout_din is equal to the weight matrix. dout_dw is to the input array.
			dpred_dout[j] = temp*(1 - temp);		//partial derivative of the post-sigmoid output with respect to the pre-sigmoid output
		}
		output[j] = temp;						//Squashing the result between o and 1 with the sigmoid function
	}
}

void training(double* dpred_dout, double* dout_dw, double* dprev, int num_neurons_n, int num_neurons_1, int num_neurons_0, double* w, double* b)
{
	double* temp_dpred_dout = (double*)calloc(num_neurons_n, sizeof(double));
	if (temp_dpred_dout == NULL) printf("failed!\n\n");
	double temp_out = 0;
	#pragma omp parallel for
	for (int k = 0; k < num_neurons_n; k++) //number of outputs
	{
		for (int j = 0; j < num_neurons_1; j++) //number of neurons in layer[n+1]
		{
			temp_out = LEARNING_RATE * dpred_dout[j] * dprev[k];										
			b[j] -= temp_out;										//updating b[n] which are the BIASSES between layer[n] and layer [n+1]
			for (int i = 0; i < num_neurons_0; i++)	//number of neurons in layer[n]
			{
				w[i*num_neurons_1 + j] -= temp_out *  dout_dw[i];				//updating w[n] which are the WEIGHTS between layer[n] and layer [n+1]	
				temp_dpred_dout[k] += temp_out*w[i*num_neurons_1 + j];		
			}
		}
	}
	#pragma omp parallel for
	for (int k = 0; k < num_neurons_n; k++)
	{
			dprev[k] = temp_dpred_dout[k];
	}

	free(temp_dpred_dout);
//	printf("");
	return;
}