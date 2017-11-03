// Parameterised Neural Net With Sigmoid.cpp : Defines the entry point for the console application.
//

#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#define LEARNING_RATE 0.5
#define TXTFILE ".txt"
#define NETWORKNUMBER ""
#define FILEPATH ".\\"

double sigmoid(double x);
void randomise(double* input, int inputcomponents);
void net_layer(double* input, int num_of_inputs, double* output, int num_of_outputs, double* w, double* b, double* dpred_dout, int training_flag);
void file_open_check(FILE* file, char* file_name);
void training(double*dpred_dout,double* dout_dw, double* dprev, int num_neurons_n, int num_neurons_1, int num_neurons_0, double* w, double* b);
int max_value(int* x, int num_values);

//defining globals
int training_flag = 0;



int main()
{
    int modval = 10;
    //////////////////////////////////////////////////////////// READING FROM INITIALISATION FILE //////////////////////////////////////////////////////////////////////////////////////////

    //Opening the initialisation file
    FILE *init_file;
    init_file = fopen(FILEPATH"NN_init" NETWORKNUMBER TXTFILE , "r");
    file_open_check(init_file, "INITIALISATION");

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
    modval = floor(100.0*exp(-0.005*data_volume)+1); // This is so that the terminal readout doesn't slow down the training too significantly, varies based on number of samples to train over.
    //////////////////////////////////////////////////////////// FINISHED READING FROM INITIALISATION FILE //////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////// READING FROM DATA FILE //////////////////////////////////////////////////////////////////////////////////////////
    //Opening the Data file
    FILE *data_file;
    data_file = fopen(FILEPATH"NN_data" NETWORKNUMBER TXTFILE, "r");
    file_open_check(init_file, "DATA");

    double* target	= (double*)malloc(sizeof(double) * num_neurons[num_layers - 1] * data_volume);		//initialising the target array (matrix) size
    double* data	= (double*)malloc(sizeof(double) * data_volume * num_neurons[0]);				//initialising the data array (matrix) size
    if (e != (num_layers))
    {
        printf("ERROR! Data in initialisation file does not have the correct number of entries!\nRead %d entries, expected %d entries.\nCheck the entries in the initialisation file is correct!",e,num_layers);
        return(0);
    }
    else printf("Initialisation completed successfully\n");
    fclose(init_file);


    e = 0; //resetting error flag

    //taking data from file and storing it in relevant information
    printf("Reading Data...\n");
    while (!feof(data_file))
    {
        //This loop grabs the data for the specific entry and stores it in the data array.
        for (int i = 0; i < num_neurons[0]; i++) fscanf(data_file, "%lf", &data[num_neurons[0] * e + i]);

        //If the neural network is getting trained, iterate through and store the target variables
        if ((training_flag == 1)|(training_flag == 2)) for(int i = 0; i < num_neurons[num_layers - 1]; i++) fscanf(data_file, "%lf", &target[num_neurons[(num_layers - 1)] * e + i]);
        e++;
        if((e+1)%10==0)
            printf("\r %7.3lf%%",((double)(e+1)/data_volume)*100);
    }
    printf("\n");
    if (e != (data_volume))
    {
        printf("ERROR! Data in data file does not have the correct number of entries!\nRead %d entries, expected %d entries.\nCheck the entires in the data file and initialisation file is correct!",e,data_volume);
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

    //////////////////////////////////////////////////////////// READING FROM WEIGHT AND BIAS FILES //////////////////////////////////////////////////////////////////////////////////////////
    //if the system is assuming an already trained neural net, open and load the weights and biasses between each layer.
    if ((training_flag==0)|(training_flag == 2))
    {
        FILE* w_file;
        w_file = fopen(FILEPATH"NN_w" NETWORKNUMBER TXTFILE, "r");
        file_open_check(init_file, "WEIGHT");

        FILE* b_file;
        b_file = fopen(FILEPATH"NN_b" NETWORKNUMBER TXTFILE, "r");
        file_open_check(init_file, "BIAS");
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
                printf("Error! Weight file has incorrect entries! Please re-train system or fix issue\nProgram has not exited\n");
                    return 0;
            }
        }
        printf("Weight values successfully read from file\n");

        e = 0;
        while(!feof(b_file))
        {
            for (int j = 0; j < num_neurons[e + 1]; j++)
            {
                if (!feof(b_file)) fscanf(b_file, "%lf\t", &b[e][j]);
            }
            e++;
            if (e == (num_layers - 1) && !feof(w_file))
            {
                printf("Error! Weight file has incorrect entries! Please re-train system or fix issue\nProgram has not exited\n");
                return 0;
            }
        }
        printf("Bias values successfully read from file\n");
        e = 0;
        fclose(w_file);
        fclose(b_file);

    }
    //////////////////////////////////////////////////////////// FINISHED READING WEIGHTS AND BIASSES //////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////		EVALUATION STATE STARTS HERE		//////////////////////////////////////////////////////////
    //In evaluation state (training_flag = 0), the system does not perform back-propagation and is only used to take inputs and make predictions on what the output should
    //be from those inputs. Input data alone with the estimates are stored in NN_eval
    if (training_flag == 0)
    {
        //opening the file the data with result will be stored in.
        FILE* eval_file;
        eval_file = fopen(FILEPATH"NN_eval" NETWORKNUMBER TXTFILE, "w");
        file_open_check(eval_file, "EVALUATION");


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
    /////////////////////////////////////////////////////////////		EVALUATION STATE FINISHES HERE		/////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////		TRAINING STATE STARTS HERE			/////////////////////////////////////////////////////////////////////
    //In training state the data is propagated forwards, where the result is then compared to target values creating a cost squared function. The partial derivative
    //of the cost function is then propagated backwards through the system, adjusting the weight bias values of each layer to improve the performance of the system.
    //Back propagation uses calculus and linear algebra to achieve this, the matrix size is optimised to improve the speed of the system by removing unecessary rows
    //and columns. ASK CAMERON FOR EXPLANATION IF YOU GET LOST
    else
    {
        FILE* cost_file;
        cost_file = fopen(FILEPATH"NN_cost" NETWORKNUMBER TXTFILE, "w");
        file_open_check(cost_file, "COST");
        fprintf(cost_file,"%lf\n",LEARNING_RATE);
        fclose(cost_file);
        int d = 0;
        //creating temporary square matrix which will hold intermediate values of the cascading partial derivative between layers
        double* dprev = (double*)calloc(max_value(num_neurons,num_layers), sizeof(double));
        double cost = 0;
        printf("Training...\n");
        for (int x = 0; x < iterations; x++)
        {
            cost = 0;
            for (int d = 0; d < data_volume; d++){
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
                    dprev[k] = 2.0 * (layer[num_layers - 1][k] - target[d*num_neurons[num_layers - 1] + k]);
                    cost += (layer[num_layers - 1][k] - target[d*num_neurons[num_layers - 1] + k])*(layer[num_layers - 1][k] - target[d*num_neurons[num_layers - 1] + k]);
                }
                for (int j = (num_layers - 1); j > 0; j--)
                {
                    training(&dpred_dout[j-1][0], &layer[j-1][0], &dprev[0], num_neurons[num_layers - 1], num_neurons[j], num_neurons[j - 1], &w[j - 1][0], &b[j - 1][0]);
                }
                /////////////////////////////////////////////////////////////		BACK PROPAGATION FINISHES HERE!		//////////////////////////////////////////////////////////
            }
            if(((x+1)%modval)==0){
                printf("\r %7.3lf%%, cost = %10.6lf",((x+1)/iterations*100),cost);
                cost_file = fopen(FILEPATH"NN_cost" NETWORKNUMBER TXTFILE, "a");
                fprintf(cost_file,"%6d   \t%10.6lf\n",x,cost);
                fclose(cost_file);
            }
        }
        printf("\nTraining Complete.\n");
        free(dprev);
        FILE *w_file;
        FILE *b_file;
        w_file = fopen(FILEPATH"NN_w" NETWORKNUMBER TXTFILE, "w");
        file_open_check(w_file, "WEIGHT");
        b_file = fopen(FILEPATH"NN_b" NETWORKNUMBER TXTFILE, "w");
        file_open_check(b_file, "BIAS");

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
    ////////////////////////////////////////////////////////////     TRAINING STATE FINISHES HERE     //////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////// (crude) TESTING STATE STARTS HERE    //////////////////////////////////////////////////////////////////////////////////////////
    FILE* eval_file;
    eval_file = fopen(FILEPATH"NN_eval" NETWORKNUMBER TXTFILE, "w");
    file_open_check(eval_file, "EVALUATION");
    double rms_error = 0;
    double tmp_error = 0;
    int x;
    printf("Testing Performance...\n");
    if (training_flag == 1) {
        int d = 0;
        for (x = 0;x<0.3*data_volume;x++) { // Pretty crappy way to do this but here we are...
            if (x>0)
                fprintf(eval_file,"\n");
            d = rand() % (int)data_volume;	// choosing a random data point to train the network
            for (int j = 0; j < num_neurons[0]; j++)
            {
                layer[0][j] = data[d*num_neurons[0] + j]; //updating the input buffer/layer
            }
            /////////////////////////////////////////////////////////////		Forward propagation					/////////////////////////////////////////////////////////////
            for (int j = 0; j < (num_layers - 1); j++)
            {
                net_layer(&layer[j][0], num_neurons[j], &layer[j + 1][0], num_neurons[j + 1], &w[j][0], &b[j][0], &dpred_dout[j][0],  0);
            }
            tmp_error = 0;
            for(int j = 0; j<num_neurons[num_layers - 1]; j++){
                tmp_error += pow(layer[num_layers - 1][j]-target[d*num_neurons[num_layers - 1] + j],2);
                fprintf(eval_file,"%10.6lf ",layer[num_layers - 1][j]);
            }
            for(int j = 0; j<num_neurons[num_layers - 1]; j++) fprintf(eval_file,"%10.6lf ",target[d*num_neurons[num_layers - 1] + j],2);
            rms_error += tmp_error/(num_neurons[num_layers-1]);
        }
        rms_error = sqrt(rms_error/x);
        printf("Testing Complete.\nRMS Error:\n%6.2lf",rms_error);
    }
    fclose(eval_file);
    ////////////////////////////////////////////////////////////     TESTING STATE FINISHES HERE      //////////////////////////////////////////////////////////////////////////////////////////


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

//////////////////////////////////////////////////////////// END OF MAIN //////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////// END OF MAIN //////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////// END OF MAIN //////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////// END OF MAIN //////////////////////////////////////////////////////////////////////////////////////////





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
        output[j] = temp;						//Squashing the result between 0 and 1 with the sigmoid function
    }
}

//The training function contains the back-propagation function within the
void training(double* dpred_dout, double* dout_dw, double* dprev, int num_neurons_n, int num_neurons_1, int num_neurons_0, double* w, double* b)
{
    double* temp_dpred_dout = (double*)calloc(num_neurons_0, sizeof(double));
    if (temp_dpred_dout == NULL) printf("failed!\n\n");
    double temp_out = 0;
    #pragma omp parallel for
    for (int j = 0; j < num_neurons_1; j++)
    {
        temp_out = LEARNING_RATE * dpred_dout[j] * dprev[j];
        b[j] -= temp_out;
        for (int i = 0; i < num_neurons_0; i++)
        {
            w[i*num_neurons_1 + j] -= temp_out *  dout_dw[i];
            temp_dpred_dout[i] += temp_out*w[i*num_neurons_1 + j];
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < num_neurons_0; i++)
    {
            dprev[i] = temp_dpred_dout[i];
    }
    free(temp_dpred_dout);
    return;
}

void file_open_check(FILE* file, char* file_name)
{
    if (file == NULL)
    {
        printf("ERROR! can't open %s file to read, check file path!\n", file_name);
        throw;
    }
    else printf("%s file successfully opened\n", file_name);
}
