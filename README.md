# NN
The initialisation file should be stored as NN_init.txt as per the NN template excel sheet, ensure that there are no empty rows when copy-pasted into the .txt file. The initialisation file will contain the following variables:
Training mode, iterations, convergence condition, convergence count, learning rate, number of layers, number of data sets, number of neurons in each layer.

TRAINING MODE
This currently has 3 different modes specified by the training mode initialisation.
Mode 0: 	Running mode, enables raw data to be used on a pre-trained NN, it will output a file called NN_eval.txt in the specified directory, which will contain the entered data set along-side the NN's prediction.
Mode 1: 	Fresh/first training mode, will randomly initialise all the weight variables between -1.0 and 1.0, where it then trains the system for the specified amount of iterations through random points in the data set.
Mode 2:		Continued training mode, enables continued training on an already trained NN, this may be useful when new data has become available where it is not desired to train the NN from scratch.

ITERATIONS
The iterations is the amount of data points you wish to use to train your data set with. If you wish to train the system via convergence instead of a set amount of iterations, the iterations value will still be used as a safety net for systems which do not converge.

CONVERGENCE CONDITION
This is the threshold for which convergence is considered (a convergence condition of 0.1 means that the cost squared, pred-target squared, is less than or equal to 0.1), if it is not wished to use this, method of training, set the convergence condition to 0.0 

CONVERGENCE COUNT
The convergence count is the amount of time the neural network successfully meets the convergence condition for random data points, at which time the program will stop training. 

LEARNING RATE
The learning rate is fraction of the partial derivative that the weights and biasses are corrected by 
e.g. w[n] = LEARNING_RATE* dout/dw[n]

NUMBER OF LAYERS
This is used to design the specific neural network size, note that the output layer and input layer are included in this variable, hence number of layers must >= 2. 

NUMBER OF DATA SETS
This is used to declare how many data sets will be uploaded into the system, for running mode or training mode 0
