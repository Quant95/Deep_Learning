from builtins import range
from builtins import object
import numpy as np
import numpy.random as npr

from cs231n.layers import *
from cs231n.layer_utils import *

#np.seterr(divide='ignore', invalid='ignore')

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        pass
        # initilize
        W1 = weight_scale*npr.normal(0, 1, (input_dim,hidden_dim) )
        b1 = np.zeros(hidden_dim)
        W2 = weight_scale*npr.normal(0, 1, (hidden_dim,num_classes) )
        b2 = np.zeros(num_classes)
        
        # store
        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        pass
        N = X.shape[0]
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        
        '''X:  (N, d_1, ..., d_k) ==> (N, D)
           W1: (D, H); b1: (H,)
           W2: (H, C); b2: (C,)
        '''
        #X = X.reshape(N,-1)
        #z1 = X@W1 + b1  # z1: (N,H)
        
        y1, cache1 = affine_relu_forward(X, W1, b1)        
        y2, cache2 = affine_relu_forward(y1, W2, b2)
        # cache1: (x, w1, b1, relu_input1);
        # cache2: (y1, w2, b2, relu_input2) 
        cache = (cache1, cache2) 
                                 
        scores = y2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        pass
        scores_exp = np.exp(scores)
        
        y_pred = np.argmax(scores,axis=1) 
        L2_term = np.sum(W1**2) + np.sum(W2**2)
        loss = -np.sum( np.log( scores_exp[np.arange(N),y]/np.sum(scores_exp,axis=1) ))/N  + 0.5*self.reg*L2_term
        
        dout = scores_exp/np.sum(scores_exp,axis=1).reshape(N,1)
        dout[np.arange(N),y] += -1 
        dy1, dW2, db2 = affine_relu_backward(dout, cache2)
        dW2 /= N
        dW2 += self.reg*W2
        db2 /= N
        dX, dW1, db1 = affine_relu_backward(dy1, cache1)
        dW1 /= N
        dW1 += self.reg*W1
        db1 /= N

        grads.update({'W1': dW1,
                      'b1': db1,
                      'W2': dW2,
                      'b2': db2})
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1  # 1 False, not use dropout
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        pass
        #hidden_num = len(hidden_dims)
        #self.params['W1'] = npr.normal(0,1, (input_dim,hidden_dims[0]) )
        #for i in range(self.num_layers):
        #    self.params['W%d' %(i+1)] = npr.randn(0, 1, (hidden_dims[i], hidden_dims[i+1]) )
        dim_layers = [input_dim] + hidden_dims + [num_classes]
        gambet_need_layers =  hidden_dims + [num_classes]
        #print("dim_layers", dim_layers)
        for i in range(self.num_layers):
            #print("weight_scale type", type(weight_scale) )
            #print("i type ", type(i))
            #print("dim_layers[i type]", type(dim_layers[i]))
            #print("dim_layers[i type]", dim_layers[i])
            self.params['W%d' %(i+1)] = weight_scale * npr.normal(0,1, (dim_layers[i], dim_layers[i+1]) )
            self.params['b%d' %(i+1)] = np.zeros((dim_layers[i+1]))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            #self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers )]
            gammas = {'gamma' + str(i+1): np.ones(gambet_need_layers[i]) 
                     for i in range(self.num_layers) }
            betas = {'beta' + str(i+1): np.zeros(gambet_need_layers[i])
                     for i in range(self.num_layers)}
            
            self.params.update(betas)
            self.params.update(gammas)
            #print('bn_params', self.bn_params)
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        pass
        out = {}
        cache = {}
        #print("########## now begin forward ")
        X = X.reshape(X.shape[0], -1) 
        if self.normalization=='batchnorm':
            #print(' garams1', self.params['gamma1'].shape)
            #print('gamma2 shape', self.params['gamma2'].shape)
            #print('gamma3 shape', self.params['gamma3'].shape)
            gamma = self.params['gamma1']
            beta = self.params['beta1']
            #print('X hspae', X.shape)
            out['y1'], cache['bn1'] = affine_bn_relu_forward(X, self.params['W1'], self.params['b1'],
                                          gamma, beta, self.bn_params[0]  )
        else:
            out['y1'], cache['y1'] = affine_relu_forward(X, self.params['W1'], self.params['b1'])
            
        if self.use_dropout:
            #print("dropout layer 1")
            #print("using dropout")
            out['y1'], cache['mask1'] = dropout_forward(out['y1'], self.dropout_param)
        
            
        #print('out[y1].shape', out['y1'].shape)
        
        #print('self.num_layers', self.num_layers)
        for i in range(self.num_layers):
            i +=1
            #print('i=', i)
            if(i==self.num_layers):
                break  
            if self.normalization=='batchnorm':
                #print('-------------------------')
                #print('num %d layer' %i)
                out['y%d' %(i+1)], cache['bn%d' %(i+1)] = affine_bn_relu_forward( out['y%d' %(i)],
                                         self.params['W%d' %(i+1)], self.params['b%d' %(i+1)],
                                         self.params['gamma%d' %(i+1)], self.params['beta%d' %(i+1)],                                            self.bn_params[i]  ) 
                
            else:
                out['y%d' %(i+1)], cache['y%d' %(i+1)] = affine_relu_forward(out['y%d' %(i)], 
                                         self.params['W%d' %(i+1)], self.params['b%d' %(i+1)])
            if self.use_dropout:
                #print('dropout layer %d' %i)
                out['y%d' %(i+1)], cache['mask%d' %(i+1)] = dropout_forward(out['y%d' %(i+1)],
                                                                           self.dropout_param)
            
        scores = out['y%d' %self.num_layers]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        grads={}
        med_val ={}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        pass
        N = X.shape[0]
        exp_scores= np.exp(scores)
        loss = np.sum( -scores[np.arange(N),y] + np.log(np.sum(exp_scores,axis=1)) )
        L2_term = 0
        for i in range(self.num_layers):
            L2_term += np.sum( self.params['W%d' %(i+1)]**2 )
        loss /= N
        loss += 0.5*self.reg*L2_term
    
        dout = exp_scores/np.sum(exp_scores,axis=1).reshape(N,1)
        dout[np.arange(N),y] += -1
        med_val['x%d' %(self.num_layers)] = dout
        
        #print("########## now begin backward #############")
        for i in range(self.num_layers, -1, -1):
            
            if(i==0):
                break
            
            if self.use_dropout:
                #print(" backward layer %d" %(i+1))
                med_val['x%d' %(i)] = dropout_backward(med_val['x%d' %(i)], cache['mask%d' %(i)])
            
            
            if self.normalization=='batchnorm':
                med_val['x%d' %(i-1)], grads['W%d' %(i)], grads['b%d' %(i)], \
                grads['gamma%d' %(i)], grads['beta%d' %(i)] = affine_bn_relu_backward(
                             med_val['x%d' %(i)], cache['bn%d' %(i)])
                grads['W%d' %(i)] /= N
                grads['W%d' %(i)] += self.reg*self.params['W%d' %(i)]
                grads['b%d' %(i)] /= N
                grads['gamma%d' %(i)] /= N
                grads['beta%d' %(i)] /= N
            else:
                med_val['x%d' %(i-1)], grads['W%d' %(i)], grads['b%d' %(i)]= \
                                   affine_relu_backward( med_val['x%d' %(i)],cache['y%d' %(i)] )
                grads['W%d' %(i)] /= N
                grads['W%d' %(i)] += self.reg*self.params['W%d' %(i)]
                grads['b%d' %(i)] /= N
                
            
                
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
    
def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    #print('in affine_bn_relu')
    #print('x shape', x.shape)
    aff_out, aff_cache = affine_forward(x, w, b)
    #print('aff_out shape', aff_out.shape)
    bn_out, bn_cache = batchnorm_forward(aff_out, gamma, beta, bn_param)
    out, ru_cache = relu_forward(bn_out)
    cache = (aff_cache, bn_cache, ru_cache)
    return out, cache
    
def affine_bn_relu_backward(dout, cache):
    aff_cache, bn_cache, ru_cache = cache
        
    drelu = relu_backward(dout, ru_cache)
    dbn, dgamma, dbeta = batchnorm_backward_alt(drelu, bn_cache)       
    dx, dw, db = affine_backward(dbn, aff_cache)
        
    return dx, dw, db, dgamma, dbeta
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        