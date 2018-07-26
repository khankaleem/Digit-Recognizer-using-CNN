'''''''''
Project: Digit recognition using sigmoid neural network using MNIST data from SCRATCH!
Kaleem Ahmad
IIT(ISM) Dhanbad
CSE
'''''''''

import BackPropogationCNN
import CNN
import numpy as np
import LoadData

ETA = 1.5
EPOCHS = 30
INPUT_SHAPE = (28*28)     # for mnist
BATCH_SIZE = 10
LMBDA = 0.1

training_data, validation_data, test_data = LoadData.ProcessData()
x,y = list(training_data)[0][0].shape
input_shape = (1,x,y)
print('shape of input data: ' +str(input_shape))

net = Model(input_shape,
            layer_config = [
                {'conv_layer': {
                    'filter_size' : 5,
                    'stride' : 1,
                    'num_filters' : 20}},
                {'pool_layer': {
                    'pool_size' : (2,2)}},
                {'fc_layer': {
                    'num_output' : 30}},
                {'final_layer': {
                    'num_classes' : 10}}
            ])

net.SGD(training_data, BATCH_SIZE, ETA, EPOCHS, test_data = list(test_data)[0:10])