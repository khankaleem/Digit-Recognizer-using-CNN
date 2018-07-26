'''
Implementation of BackPropogation for CNN

delta = del(Loss)/del(z_val)
'''
import numpy as np
'''
From Fully Connected layerto Fully Connected layer
'''
def BackProp_FC_to_FC(delta, prev_weights, prev_activations, z_vals, final = False):
    '''
    Reset delta if not FC is not a final layer 
    '''
    if not final:
        sp = sigmoid_prime(z_vals)
        delta = np.dot(prev_weights.transpose(), delta) * sp
        
    Slopeb = delta
    Slopew = np.dot(delta, prev_activations.transpose())
    return (Slopew, Slopeb, delta)

'''
From fully connected to 3D layer
'''
def BackProp_FC_TO_3D(delta, prev_weights, prev_activations, z_vals):
    #Calculate delta
    sp = sigmoid_prime(z_vals)
    delta = np.dot(prev_weights.transpose(), delta) * sp
    
    Slopeb = delta
    images
    #Reshape Prev activations
    depth, dim1, dim2  = prev_activations.shape
    prev_activations = prev_activations.reshape((1, depth * dim1 * dim2))
    
    Slopew = np.dot(delta, prev_activations)
    Slopew = Slopew.reshape((delta.shape[0], depth, dim1, dim2))
    
    return (Slopew, Slopeb, delta)

'''
From Pooling layer to convolution layer
Only delta Changes
'''
def BackProp_Pool_to_Conv(delta, prev_weights, input_from_conv, max_indices, pool_size, pool_output):
    #Get Shape
    x, y, z = pool_output.shape
    a, b, c, d = prev_weights.shape
    
    #Reshape Weights and PoolLayer
    prev_weights = prev_weights.reshape((a, b * c * d))
    pool_output = pool_output.reshape((x * y * z, 1))

    #Reshape MaxIndex matrix
    max_indices = max_indices.reshape((x, y * z, 2))
    
    #Bckpropogate delta from fc layer to pooLLayer
    sp = sigmoid_prime(pool_output)
    delta = np.dot(prev_weights.transpose(), delta) * sp
    delta = delta.reshape((x, y * z))
    pool_output = pool_output.reshape((x, y * z))
    
    #depth height width
    depth, height, width = input_from_conv.shape
    delta_new = np.zeros((depth, height, width))
    
    for d in range(depth):
        row, col = 0, 0
        for i in range(max_indices.shape[1]):
            to_pool = input_from_conv[d][row : poolsize[0] + row, col : poolsize[1] + col]
            
            #Get new delta
            delta_from_pooling = max_prime(pool_output[d][i], delta[d][i], to_pool)
            delta_new[d][row : poolsize[0] + row, col : poolsize[1] + col] = delta_from_pooling
            
            col += poolsize[1]
            if col >= width:
                col = 0
                row += poolsize[1]
                
    #Return New delta 
    return delta_new

def BackProp_To_Conv(delta, weight_filters, stride, input_to_conv, pre_z_values):
    #Get shape of weights
    num_filters, depth, filter_size, filter_size = weight_filters.shape

    #Initialze Slope of weights and biases
    Slopeb = np.zeros((num_filters, 1))
    Slopew = np.zeros((weight_filters.shape))
    
    total_delta_per_layers =  delta.shape[1]*delta.shape[2]
    #Reshape delta
    delta = delta.reshape((delta.shape[0], total_delta_per_layers))
    
    #For all Filters evaluate Slopew and Slopeb for filter
    for j in range(num_filters):
        col, row = 0, 0
        for i in range(total_delta_per_layers):
            to_conv = input_to_conv[:, row:row+filter_size, col:col+filter_size]
            Slopew[j] += to_conv * delta[j][i]
            Slopeb[j] += delta[j][i]
            #update col
            col += stride[1]
                
            
            if (col + filter_size) - stride >= input_to_conv.shape[2]:
                col = 0
                row += stride
    return (Slopeb, Slopew)

#Helper functon for transition from pooling layer to convolution layer
def max_prime(res, delta, tile_to_pool):
    dim1, dim2 = tile_to_pool.shape
    tile_to_pool = tile_to_pool.reshape((dim1 * dim2))
    new_delta = np.zeros((tile_to_pool.shape))
    
    for i in range(len(tile_to_pool)):
        num = tile_to_pool[i]
        if num < res:
            new_delta[i] = 0
        else:
            new_delta[i] = delta
    return new_delta.reshape((dim1, dim2))
