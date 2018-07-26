'''
This Module Loads MNIST data
and wraps it in the desired
form.  
'''

import pickle
import gzip
import numpy as np

def GetData():
    file = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(file, encoding='latin1')
    file.close()
    return (training_data, validation_data, test_data)

def GetOutput(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e    
    
def ProcessData():
    '''
    tr_d, va_d, te_d = GetData()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [GetOutput(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)
    '''
    tr_d, va_d, te_d = GetData()
    training_inputs = [np.reshape(x, (28, 28)) for x in tr_d[0]]
    training_results = [GetOutput(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (28, 28)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (28, 28)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)
    