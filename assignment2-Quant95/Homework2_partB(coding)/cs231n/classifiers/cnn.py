from builtins import object
import numpy as np
import numpy.random as npr

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 #conv_param={'stride':1, 'pad':1}, 
                 #pool_param={'pool_height': 1, 'pool_weight':1, 'stride':1},
                 hidden_dim=100, num_classes=10,  weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #                           
        ############################################################################
        pass
        # x: (N, C, H, W)
        # w1 (filter): (F, C, HH, WW)
        # b1: (F,)
        # out1: (N, F, Hh, Ww) ==> reshape (N, D)
        # 
        # w2: (D, M), M = hidden_dim
        # b2: (M,)
        #
        # W3: (M, output)
        # b3: (output,)
        C, H, W = input_dim
        F = num_filters
        HH = filter_size
        WW = filter_size
        #pad = conv_param['pad']
        pad = (filter_size - 1) // 2
        #s_conv = conv_param['stride']
        s_conv = 1
        #ph = pool_param['pool_height']
        #pw = pool_param['pool_weight']
        #s_pool = pool_param['stride']
        ph, pw, s_pool = 2, 2, 2
        
        
        
        W1 = weight_scale * npr.normal(0, 1, (F,C,HH,WW))
        b1 = np.zeros((F))
        
        Hh_prime = int( (H+pad*2-HH)/s_conv + 1 )
        Ww_prime = int( (W+pad*2-WW)/s_conv + 1 )
        Hh = int( (Hh_prime-ph)/s_pool + 1 )
        Ww = int( (Ww_prime-pw)/s_pool + 1 )
        W2 = weight_scale * npr.normal(0,1, ( F*Hh*Ww, hidden_dim) )
        b2 = np.zeros( (hidden_dim) )
        
        W3 = weight_scale * npr.normal(0,1, (hidden_dim, num_classes) )
        b3 = np.zeros( (num_classes) )
        
        self.params.update({'W1': W1,
                            'b1': b1,
                            'W2': W2,
                            'b2': b2,
                            'W3': W3,
                            'b3': b3})
        #self.conv_param = conv_param
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        pass
        
        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out2, cache2 = affine_relu_forward(out1, W2, b2)
        out3, cache3 = affine_forward(out2, W3, b3)
        
        scores = out3
  
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        pass
        N = X.shape[0]
               
        scores_exp = np.exp(scores)
        
        y_pred = np.argmax(scores,axis=1) 
        L2_term = np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2)
        loss = -np.sum( np.log( scores_exp[np.arange(N),y]/np.sum(scores_exp,axis=1) ))/N  + \
                   0.5*self.reg*L2_term
        
        dout = scores_exp/np.sum(scores_exp,axis=1).reshape(N,1)
        dout[np.arange(N),y] += -1 
        
        dout3 = dout
        dout2, dW3, db3 = affine_backward(dout3, cache3)
        dW3 /= N
        dW3 += self.reg*W3
        db3 /= N
        
        dout1, dW2, db2 = affine_relu_backward(dout2, cache2)
        dW2 /= N
        dW2 += self.reg*W2
        db2 /= N
        
        dx, dW1, db1 = conv_relu_pool_backward(dout1, cache1)
        dW1 /= N
        dW1 += self.reg*W1
        db1 /= N
        
        grads.update({'W1': dW1,
                      'b1': db1,
                      'W2': dW2,
                      'b2': db2,
                      'W3': dW3,
                      'b3': db3})
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    