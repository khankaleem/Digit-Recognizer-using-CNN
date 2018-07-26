'''
Classes for various layers in CNN
'''

#Convolution Layer
''''
To summarize, the Conv Layer:
Accepts a volume of size W1×H1×D1
Requires four hyperparameters:
Number of filters K,
their spatial extent F,
the stride S,
the amount of zero padding P
Produces a volume of size W2×H2×D2
where:
W2 = (W1−F+2P)/S+1
H2 = (H1−F+2P)/S+1
D2 = K
'''
import numpy as np
import random
class ConvLayer(object):
    def __init__(self, input_shape, filter_size, stride, num_filters, padding = 0):
        self.depth, self.height_in, self.width_in = input_shape
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.num_filters = num_filters
        
        #Initialise weight matrix and biases
        self.weights = np.random.randn(self.num_filters, self.depth, self.filter_size, self.filter_size) 
        self.biases = np.random.randn(self.num_filters, 1)#Biases are same for every depth matrix
        
        #Initialze output dimensions
        self.output_dim1 = (self.height_in - self.filter_size + 2*padding)//self.stride + 1
        self.output_dim2 = (self.width_in - self.filter_size + 2*padding)//self.stride + 1

        #Initialize output dimensions
        self.z_values = np.zeros((self.num_filters, self.output_dim1, self.output_dim2))
        self.output = np.zeros((self.num_filters, self.output_dim1, self.output_dim2))        
        
    def Convolve(self, input_neuron):
        '''
        Convolute the input_neuron with weight matrix
        to produce sigmoid activation matrix for next layer
        '''
        #reshape to a linear array
        self.z_values = self.z_values.reshape((self.num_filters, self.output_dim1 * self.output_dim2))
        self.output = self.output.reshape((self.num_filters, self.output_dim1 * self.output_dim2))
        act_length1d =  self.output.shape[1]

        for j in range(self.num_filters):
            col = 0
            row = 0

            for i in range(act_length1d):
                #loop til the output array is filled up -> one dimensional (600)
                # ACTIVATIONS -> loop through each conv block horizontally
                self.z_values[j][i] = np.sum(input_neuron[:,row:self.filter_size+row, col:self.filter_size + col] * self.weights[j]) + self.biases[j]
                self.output[j][i] = sigmoid(self.z_values[j][i])
                col += self.stride

                if (self.filter_size + col)-self.stride >= self.width_in:
                    #wrap indices at the end of each row
                    col = 0
                    row += self.stride
        
        #Restore Shape of output matrix   
        self.z_values = np.zeros((self.num_filters, self.output_dim1, self.output_dim2))
        self.output = np.zeros((self.num_filters, self.output_dim1, self.output_dim2))

''''
Accepts a volume of size W1×H1×D1
Requires two hyperparameters:
their spatial extent F,
the stride S,
Produces a volume of size W2×H2×D2
where:
W2=(W1−F)/S+1
H2=(H1−F)/S+1
D2=D1
'''                
#Pooling Layer
class PoolingLayer(object):
    def __init__(self, input_shape, pool_size = (2, 2)):
        self.depth, self.height_in, self.width_in = input_shape
        self.pool_size = pool_size
        
        #Initilize otput heights and width
        self.height_out = (self.height_in - self.pool_size[0])//self.pool_size[1] + 1
        self.width_out =  (self.width_in - self.pool_size[0])//self.pool_size[1] + 1
        
        #Initialize output matrix and store the indices
        self.output = np.empty((self.depth, self. height_out, self.width_out))
        self.max_indices = np.empty((self.depth, self. height_out, self.width_out, 2))
        
    def Pool(self, input_image):
        self.pool_length1d = self.height_out * self.width_out

        self.output = self.output.reshape((self.depth, self.pool_length1d))
        self.max_indices = self.max_indices.reshape((self.depth, self.pool_length1d, 2))        
        
        #for each filter map
        for i in range(self.depth):
            row, col = 0, 0
            for j in range(self.pool_length1d):
                to_pool = input_image[i][row:row+self.pool_size[0], col:col+self.pool_size[0]]
                
                self.output[i][j] = np.amax(to_pool)#store the max activation value
                index = zip(*np.where(np.max(to_pool) == to_pool))
                index = list(index)
                if len(list(index)) > 1:
                    index = [index[0]]
                    
                index = index[0][0] + row, index[0][1] + col
                self.max_indices[i][j] = index
                col += self.pool_size[1]
                
                if col >= self.width_in:                    
                    col = 0
                    row += self.pool_size[1]

        self.output = self.output.reshape((self.depth, self.height_out, self.width_out))
        self.max_indices = self.max_indices.reshape((self.depth, self.height_out, self.width_out, 2))
  
class Layer(object):

    def __init__(self, input_shape, num_output):
        self.output = np.ones((num_output, 1))
        self.z_values = np.ones((num_output, 1))
        
        
class FullyConnectedLayer(Layer):
    '''
    Calculates outputs on the fully connected layer then forwardpasses to the final output -> classes
    '''
    def __init__(self, input_shape, num_output):
        super(FullyConnectedLayer, self).__init__(input_shape, num_output)
        self.depth, self.height_in, self.width_in = input_shape
        self.num_output = num_output

        self.weights = np.random.randn(self.num_output, self.depth, self.height_in, self.width_in)
        self.biases = np.random.randn(self.num_output,1)
        
    def feedforward(self, a):
        '''
        forwardpropagates through the FC layer to the final output layer
        '''
        # roll out the dimensions
        self.weights = self.weights.reshape((self.num_output, self.depth * self.height_in * self.width_in))
        a = a.reshape((self.depth * self.height_in * self.width_in, 1))

        # this is shape of (num_outputs, 1)
        self.z_values = np.dot(self.weights, a) + self.biases
        self.output = sigmoid(self.z_values)
        self.weights = self.weights.reshape((self.num_output, self.depth, self.height_in, self.width_in))

class ClassifyLayer(Layer):
    def __init__(self, num_inputs, num_classes):
        super(ClassifyLayer, self).__init__(num_inputs, num_classes)
        num_inputs, col = num_inputs
        self.num_classes = num_classes
        self.weights = np.random.randn(self.num_classes, num_inputs)
        self.biases = np.random.randn(self.num_classes,1)
        
    def Classify(self, x):
        self.z_values = np.dot(self.weights,x) + self.biases
        self.output = sigmoid(self.z_values)
        
     
#CNN Model
class Model(object):
    layer_type_map = {
        'fc_layer': FullyConnectedLayer,
        'final_layer': ClassifyLayer,
        'conv_layer': ConvLayer,
        'pool_layer': PoolingLayer
    }
    
    def __init__(self, input_shape, layer_config):
        self.input_shape = input_shape
        self.Initialize_Layers(layer_config)#Initialize layers of the network
        #Store Layer Shapes of weights and biases
        self.layer_weight_shapes = [l.weights.shape for l in self.layers if not isinstance(l, PoolingLayer)]
        self.layer_biases_shapes = [l.biases.shape for l in self.layers if not isinstance(l, PoolingLayer)]
        
    #Initialize layers
    def Initialize_Layers(self, layer_config):
        layers = []
        input_shape = self.input_shape
        for layer_spec in layer_config:
            layer_class = self.layer_type_map[list(layer_spec.keys())[0]]
            layer_kwargs = list(layer_spec.values())[0]
            layer = layer_class(input_shape, **layer_kwargs)
            input_shape = layer.output.shape
            layers.append(layer)
        self.layers = layers
        
    #Get Transition of layer    
    def GetTransition(self, inner_layer, outer_layer):
        inner, outer = self.layers[inner_ix], self.layers[outer_ix]
        if isinstance(inner, FullyConnectedLayer) and (isinstance(outer, ClassifyLayer) or isinstance(outer, FullyConnectedLayer)):
            return '1d_to_1d'
        elif isinstance(inner, PoolingLayer) and isinstance(outer, FullyConnectedLayer):
            return '3d_to_1d'
        elif isinstance(inner, ConvLayer) and isinstance(outer, ConvLayer):
            return 'to_conv'
        elif isinstance(inner, ContemporaryvLayer) and isinstance(outer, PoolingLayer):
            return 'conv_to_pool'
        else:
            raise NotImplementedError
    
    def FeedForward(self, image):
        prev_activation = image
        #Traverse over every Layer
        for layer in self.layers:
            input_to_feed =  prev_activation
            #Fully Connected Layer
            if isinstance(layer, FullyConnectedLayer):
                #FeedForward
                layer.feedforward(input_to_feed)
            #Convolution Layer
            elif isinstance(layer, ConvLayer):     
                #Convolve input_to_feed
                layer.Convolve(input_to_feed)                
            #Pooling Layer    
            elif isinstance(layer, PoolingLayer):
                #Pool input_to_feednumber
                layer.Pool(input_to_feed)
            #Classification Layer    
            elif isinstance(layer, ClassifyLayer):    
                #Classify input_to_feed
                layer.Classify(input_to_feed)                
            else:
                raise NotImplementedError
            prev_activation = layer.output
        
        final_activation = prev_activation
        return final_activation

    def Backpropogate(self, image, label):
        Slopew = [np.zeros(shape) for shape in self.layer_weight_shapes]
        Slopeb = [np.zeros(shape) for shape in self.layer_biases_shapes]
        
        #Get delta for final layer
        final_output = self.layers[-1].output
        last_delta = (final_output - label) * sigmoid_prime(self.layers[-1].z_values)
        last_weights = None
        final = True
        
        num_layers = len(self.layers)
        
        for l in range(num_layers - 1, -1, -1):
            inner_layer_ix = l - 1
            if l - 1 < 0:
                inner_layer_ix = 0
            outer_layer_ix = l
            
            layer = self.layers[outer_layer_ix]
            activation = self.layers[inner_layer_ix].ouput if inner_layer_ix >= 0 else image

            transition = self.GetTransition(inner_layer_ix, outer_layer_ix)
            
            if transition == '1d_to_1d':
                dw, db, last_delta = BackProp_FC_to_FC(
                    delta = last_delta, 
                    prev_weights = last_weights, 
                    prev_activatiotemporaryns = activation, 
                    z_vals = self.layers[inner_layer_ix].z_values, 
                    final = False)
                final = False
                
            elif transition == '3d_to_1d':
                dw, db, last_delta = BackProp_FC_TO_3D(
                    delta = last_delta, 
                    prev_weights = last_weights, 
                    prev_activations = activation, 
                    z_vals = self.layers[inner_layer_ix].z_values)
                
            elif transition == 'conv_to_pool':
                last_delta = BackProp_Pool_to_Conv(
                    delta = last_delta,
                    prev_weights = last_weights,
                    input_from_conv = activation,
                    max_indices = layer.max_indices,
                    poolsize = layer.poolsize,
                    pool_output = layer.output)
                
            elif transition == 'to_conv':
                db, dw = BackProp_To_Conv( 
                    delta = last_delta,
                    weight_filters = last_weights,
                    stride = layer.stride,
                    input_to_conv = activation,
                    prev_z_vals = layer.z_values)
                
            else:
                pass
            
            if transition != 'conv_to_pool':
                SLopew[outer_layer_ix], SLopeb[outer_layer_ix] = db, dw
                last_weights = layer.weights
            
        return self.layers[-1].output, SLopeb, Slopew

    #Stochiastic Gradient Descent
    def SGD(self, training_data, batch_size, eta, epochs, test_data = None):
        training_data = list(training_data)
        if test_data:
            test_data = list(test_data)  
            
        print('Starting Epoch................')
        for epoch in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k+batch_size] for k in range(0, len(training_data), batch_size)]
            for batch in batches:
                self.UpdateNetwork(batch, eta)
            if test_data:
                print('Epoch ' + str(epoch+1) + ': ' + str(self.Evaluate(test_data)))
            else:
                print('Epoch ' + str(epoch+1))
                
    #Update CNN taking one batch at a Time          
    def UpdateNetwork(self, batch, eta):
        Slopeb = [np.zeros(shape) for shape in self.layer_biases_shapes] 
        Slopew = [np.zeros(shape) for shape in self.layer_weight_shapes]
        for image, label in batch:
            #Feed Forward
            FeedForward(image)
            #Backpropogate to get slopes
            del_Slopeb, del_Slopw = self.Backpropogate(image, label)
            Slopew = [w+del_w for w, del_w in zip(Slopew, del_Slopew)]
            Slopeb = [b+del_b for b, del_b in zip(Slopeb, del_Slopeb)]
        
        num = 0
        weight_index = []
        for layer in self.layers:
            if not isinstance(layer,PoolingLayer):
                weight_index.append(num)
            num += 1       

        for i, (del_w, del_b) in enumerate(zip(Slopeb, Slopew)):
           layer = self.layers[weight_index[i]]
           layer.weights -= eta*del_w/len(batch)
           layer.biases -= eta*del_b/len(batch)
    
    def Evaluate(self,data):
        data = [(im.reshape((1,28,28)),y) for im,y in data]
        test_results = [(np.argmax(self.FeedForward(x)),y) for x, y in data]
        return sum(int(x == y) for x, y in test_results)

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))

def loss(desired,final):
    return 0.5*np.sum(desired-final)**2    