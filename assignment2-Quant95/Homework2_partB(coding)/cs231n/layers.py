from builtins import range
import numpy as np
import numpy.random as npr


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    pass
    N, *d_dim = x.shape
    D = np.prod(d_dim)
    #print('in affine_forward ############')
    #print('x shape', x.shape)
    #print('D', D)
    X = x.reshape(N,-1)
    #print('after x shape', x.shape)
    #print('w shape', w.shape)
    out = X@w+b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    pass
    N, *d_dim = x.shape
    D, M = w.shape
    
    dw = x.reshape(N,D).T@dout
    dx = (dout@w.T).reshape(N,*d_dim)
    db = dout.T@np.ones(N,)
 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    pass
    out = np.maximum(x,0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    pass
    dx = dout
    dx[cache<0] =0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        pass
        #batchsize = 256
        #if N > batchsize:
        #    idx = npr.choice(N, batchsize, repalce=False) # not repeat
        #    x_input = x[idx,:]
        #else:
        #    x_input = x
        x_sample = x
        #print('x_sample', x_sample.shape)
        sample_mean = np.mean(x_sample,axis=0)
        sample_var = np.std(x_sample,axis=0)**2
        
        x_input_normal = (x_sample-sample_mean)/np.sqrt(sample_var+eps)  # X_input_normal: (N,D)
        assert x_input_normal.shape==x.shape
        #print('###### in batchnorm_forward')
        #print('gamma shape', gamma.shape)
        #print('beta shape ', beta.shape)
        #print('x_input_normal shape', x_input_normal.shape)
        y = gamma * x_input_normal + beta # y: (N,D)
        assert y.shape == x.shape
        
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
        out = y
        cache = (gamma, eps, sample_mean, sample_var, x, x_input_normal)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        
        
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        pass
        mean = running_mean
        var = running_var
        
        x_normal = (x-mean)/(np.sqrt(var+eps))
        assert x_normal.shape == x.shape
        y = gamma * x_normal + beta 
        assert y.shape == x.shape
        
        out = y
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    pass
    gamma, eps, sample_mean, var, x, x_hat = cache #eps
    
    dbeta = np.sum(dout, axis = 0) # dbeta: (D,)
    
    dgamma = np.sum(dout*x_hat, axis =0) # dgamma: (D,)
    assert dgamma.shape == gamma.shape
    
    N,D=x.shape
    #dvar = np.sum((x-mean), axis=0)/N   #dvar: (D,)  *2
    #a = 1/(var+eps)**2  # *1/2  #a: (D,)
    #dx = dout
    dx_hat = dout*gamma #(N,D)
    dvar = np.sum( (-1/2)*dx_hat*(x-sample_mean)*np.power((var+eps),-3/2), axis=0 ) #(D,)
    dmean = np.sum( -dx_hat*np.power((var+eps),-1/2), axis=0 ) + \
            dvar*(-2)*np.sum(x-sample_mean,axis=0)/N  #(D,)
    dx = dx_hat*np.power((var+eps),-1/2) + 2*dvar*(x-sample_mean)/N + \
         dmean/N #(N,D)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    pass
    gamma, eps, sample_mean, var, x, x_hat = cache #eps
    N,D = x.shape
    
    dbeta = np.sum(dout, axis = 0) # dbeta: (D,)
    
    dgamma = np.sum(dout*x_hat, axis =0) # dgamma: (D,)
    
    #dL_y = dout*gamma
    
    #sigma = np.power((var+eps), 1/2)
    
    #term1 =np.sum(dout*gamma*(x-sample_mean)*(-1/2)*np.power(sigma,-3), axis=0) * \
    #       (2*(x-sample_mean)/N)
    
    #dy_mean = - 1/sigma + (1/sigma**2)*(1/sigma)*np.sum((x-sample_mean),axis=0)/N
    #dmean_x = 1/N
    #term2 = dy_mean*dmean_x
    
    #dy_x = 1/sigma
    #term3 = dy_x
    
    #dx = term1 + dL_y*(term2+term3)
    sigma = np.power((var+eps), -1/2)
    dx = (1. / N) * gamma * sigma * (N * dout - np.sum(dout, axis=0) \
          - (x - sample_mean) * (sigma**2) * np.sum(dout * (x - sample_mean), axis=0))
    
    
    #dx = dL_y* (term1+term2+term3)
    
   
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    pass
    
    N, D = x.shape
    mean_ly = np.mean(x, axis=1) # (N,)
    var_ly =  np.std(x, axis=1)**2 #(N,)
    
        
    x_input_normal = (x-mean_ly.reshape(N,1))/np.sqrt(var_ly.reshape(N,1)+eps)  # X_input_normal: (N,D)
        
    #print('in layernorm_forward shape', x_input_normal.shape)
    #print('gamma shape', gamma.shape)
    y = gamma * x_input_normal + beta # y: (N,D)
    #assert y.shape == x.shape
        
        
        
    out = y
    cache = (gamma, eps, mean_ly, var_ly, x, x_input_normal)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    pass
    gamma, eps, ln_mean, ln_var, x, x_hat = cache

    dbeta = np.sum(dout, axis = 0) # dbeta: (D,)
    
    dgamma = np.sum(dout*x_hat, axis =0) # dgamma: (D,)
    
    N,D=x.shape
    dx_hat = dout*gamma #(N,D)
    dvar = np.sum( (-1/2)*dx_hat*(x-ln_mean.reshape(N,1)) * \
                  np.power((ln_var.reshape(N,1)+eps),-3/2), axis=1 ) #(N,)      
    dmean = np.sum( -dx_hat*np.power((ln_var.reshape(N,1)+eps),-1/2), axis=1 ) + \
            dvar*(-2)*np.sum(x-ln_mean.reshape(N,1),axis=1)/D  #(N,)     
    dx = dx_hat*(np.power((ln_var.reshape(N,1)+eps),-1/2).reshape(N,1)) + \
             2*dvar.reshape(N,1)*(x-ln_mean.reshape(N,1))/D +  dmean.reshape(N,1)/D #(N,D)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        pass
        mask = (npr.rand(*x.shape)< p )/p
        y = x*mask
        
        out = y
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        pass
        mask = None
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        pass
        dx = dout*mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    pass
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    x_pad = np.pad(x, ((0,),(0,),(pad,),(pad,)),'constant')
   
    Hh = int(1 + (H+2*pad-HH)/stride)
    Ww = int(1 + (W+2*pad-WW)/stride)
   
    
    out = np.zeros((N,F,Hh,Ww))
    for i in range(N):
        
        for j in range(F):
            
            for hh in range(Hh):
                for ww in range(Ww):
                    
                    out[i,j, hh,ww] = np.sum(
                        x_pad[i, : , 0+hh*stride:0+hh*stride+HH, 0+ww*stride:0+ww*stride+WW] 
                        * w[j,:]) + b[j]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    pass
    '''
    N, F, Hh, Ww = dout.shape
    x, w, b, conv_param = cache
    _, C, H, W = x.shape
    _, _, HH, WW = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']
    
    # first calculate dx
    all_ite = N*H*W*F*Hh*Ww
    verbose = int(all_ite/50)
    ite =0
    dx = np.zeros((N, C, H, W))
    for n in range(N):
        ite +=1
        if(ite==verbose):
            print('i am in NNNNNNNN')
        for i in range(H):
            ite +=1
            if(ite==verbose):
                print('i am in HHHHHHHHH')
            for j in range(W):
                ite +=1
                if(ite==verbose):
                    print('i am in WWWWWWWWWW')
                for f in range(F):
                    ite +=1
                    if(ite==verbose):
                        print('i am in FFFFFFFFF')
                    for hh in range(Hh):
                        ite +=1
                        if(ite==verbose):
                            print('i am in hhhhhhhhhh')
                        #print('hh', hh)
                        for ww in range(Ww):
                            #print('ww', ww)
                            ite +=1
                            if(ite==verbose):
                                print('i am in wwwwwwwwww')
                            mask1 = np.zeros_like(w[f,:,:,:])
                            mask2 = np.zeros_like(w[f,:,:,:])
                            
                            if (i-stride*hh+pad)>=0 and (i-stride*hh+pad)<HH:
                                mask1[:, i-stride*hh+pad, :] = 1.0
                            if (j-stride*ww+pad)>=0 and (j-stride*ww+pad)<WW:
                                mask2[:, :, j-stride*ww+pad] = 1.0
                            dw_mask = np.sum(w[f,:,:,:]*mask1*mask2, axis=(1,2))
                            dx[n, :, i,j] += dout[n,f,hh,ww] * dw_mask
     
    
    # now calculate dw
    dw = np.zeros((F, C, HH, WW))
    x_pad = np.pad(x, ((0,),(0,),(pad,),(pad,)), 'constant')
    for f in range(F):
        for c in range(C):
            for hh in range(HH):
                for ww in range(WW):
                    mask_xpad = x_pad[:, c, hh:hh+Hh*stride:stride, ww:ww+Ww*stride:stride]
                    dw[f, c, hh, ww] = np.sum(dout[:,f,:,:]*mask_xpad)
                     
                        
    # calculate db: (F,)
    db = np.zeros((F,))
    for f in range(F):
        db[f] = np.sum(dout[:,f,:,:])
    '''
    x, w, b, conv_param = cache
  
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)
  
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
  
    db = np.sum(dout, axis = (0,2,3))
  
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    for i in range(H_out):
        for j in range(W_out):
            x_pad_masked = x_pad[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
            for k in range(F): #compute dw
                dw[k ,: ,: ,:] += np.sum(x_pad_masked * 
                                         (dout[:, k, i, j])[:, None, None, None], axis=0)
            for n in range(N): #compute dx_pad
                dx_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += np.sum((w[:, :, :, :] * 
                                                 (dout[n, :, i, j])[:,None ,None, None]), axis=0)
    dx = dx_pad[:,:,pad:-pad,pad:-pad]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    pass
    N, C, H, W = x.shape

    ph = pool_param['pool_height']
    pw = pool_param['pool_width']
    s = pool_param['stride']
    
    Hprime =int( 1 + (H - ph) / s )
    Wprime =int( 1 + (W - pw) / s )
    
    out = np.zeros((N, C, Hprime, Wprime))
    for n in range(N):
        for c in range(C):
            for i in range(Hprime):
                for j in range(Wprime):
                    out[n,c,i,j] = np.max( x[n,c, i*s:i*s+ph, j*s:j*s+pw] )
            
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    pass
    x, pool_param = cache
    N, C, H, W = x.shape
    
    ph = pool_param['pool_height']
    pw = pool_param['pool_width']
    s = pool_param['stride']
    
    Hprime =int( 1 + (H - ph) / s )
    Wprime =int( 1 + (W - pw) / s )
    
    dx = np.zeros(x.shape)
    for n in range(N):
        for c in range(C):
            for i in range(Hprime):
                for j in range(Wprime):
                    max_pooling = x[n,c, i*s:i*s+ph, j*s:j*s+pw]
                    mask = ( max_pooling==np.max(max_pooling) )
                    dx[n,c, i*s:i*s+ph, j*s:j*s+pw] = dout[n,c, i, j] * mask
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    pass
    '''
    N, C, H, W = x.shape
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    
    running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))
    
    if mode == 'train':
        x_sample = x
        sample_mean = ( 1./(N*H*W) * np.sum(x_sample, axis=(0,2,3)) ).reshape(1,C,1,1)
        sample_var = (1. / (N*H*W) * np.sum((x_sample-sample_mean)**2, 
                                            axis=(0,2,3)) ).reshape(1,C,1,1)
        
        x_input_normal = (x_sample-sample_mean)/np.sqrt(sample_var+eps)
        y = gamma.reshape(1,C,1,1,) * x_input_normal + beta.reshape(1,C,1,1)
        
        running_mean = momentum * running_mean + (1 - momentum) * np.squeeze(sample_mean)
        running_var = momentum * running_var + (1 - momentum) * np.squeeze(sample_var)     
        
        out = y
        cache = (gamma, eps, sample_mean, sample_var, x, x_input_normal)

    elif mode == 'test':
        mean = running_mean.reshape(1,C,1,1)
        var = running_var.reshape(1,C,1,1)
        
        x_normal = (x-mean)/(np.sqrt(var+eps))
        
        y = gamma.reshape(1,C,1,1) * x_normal + beta.reshape(1,C,1,1) 
        
        
        out = y
        
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    '''
    N, C, H, W = x.shape
    temp_output, cache = batchnorm_forward(x.transpose(0,3,2,1).reshape((N*H*W,C)), 
                                           gamma, beta, bn_param)
    out = temp_output.reshape(N,W,H,C).transpose(0,3,2,1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    pass
    N,C,H,W = dout.shape
    dx_temp, dgamma, dbeta = batchnorm_backward_alt(dout.transpose(0,3,2,1).reshape((N*H*W,C)),cache)
    dx = dx_temp.reshape(N,W,H,C).transpose(0,3,2,1)
    
    '''
    gamma, eps, sample_mean, var, x, x_hat = cache
   
    
    dbeta = np.sum(dout, axis=(0,2,3))
    dgamma = np.sum(dout*x_hat, axis=(0,2,3))
    
    N, C, H, W = x.shape
    Nt = N * H * W
    
    dx = (1. / Nt) * gamma.reshape(1,C,1,1) * (var + eps)**(-1. / 2.) * (
        Nt * dout
        - np.sum(dout, axis=(0, 2, 3)).reshape(1, C, 1, 1)
        - (x - sample_mean) * (var + eps)**(-1.0) * np.sum(dout * (x - sample_mean),
                                                           axis=(0, 2, 3)).reshape(1, C, 1, 1))
    '''
 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    pass
    N, C, H, W = x.shape
    x = x.reshape(N*G, C//G*H*W).T
    
    mu = np.mean(x, axis=0)
    
    xmu = x- mu
    sq = xmu**2
    var = np.var(x,axis=0)
    
    sqrtvar = np.sqrt(var+eps)
    ivar = 1./sqrtvar
    xhat = xmu*ivar
    
    xhat = np.reshape(xhat.T, (N,C,H,W))
    out = gamma[np.newaxis, :, np.newaxis, np.newaxis] * xhat + beta[np.newaxis, :,
                                                                     np.newaxis, np.newaxis]
    cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps, G)


    '''
    N, C, H, W = x.shape
    g_num = C//G   # one group has num feature
    
    x_input = x[:, 0:g_num, :,:]
    gamma = np.squeeze(gamma)
    beta = np.squeeze(beta)
   
    out, C_prime_cache = spatial_batchnorm_forward(x_input, gamma[0:g_num], beta[0:g_num], gn_param)
   
    for i in range(G):
       
        if (i==(G-1)):
            break
        i += 1
        x_input = x[:, i*g_num:(i+1)*g_num, :,:]     
        _, C_prime, _,_  = x_input.shape
        
        C_prime_out, C_prime_cache = spatial_batchnorm_forward(x_input, gamma[i*g_num:(i+1)*g_num], 
                                                               beta[i*g_num:(i+1)*g_num], gn_param)
       
        out = np.concatenate((out,C_prime_out), axis=1)
    '''    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    pass
    N, C, H, W = dout.shape
    
    xhat, gamma, xmu, ivar, sqrtvar, var, eps, G = cache
    
    dxhat = dout * gamma[np.newaxis, :, np.newaxis, np.newaxis]
    
    dgamma = np.sum(dout*xhat, axis=(0,2,3), keepdims=True)
    dbeta = np.sum(dout, axis=(0,2,3), keepdims=True)
    
    
    dxhat = np.reshape(dxhat, (N*G, C//G*H*W)).T
    xhat = np.reshape(xhat, (N*G, C//G*H*W)).T
    
    Nprime, Dprime = dxhat. shape
    
    dx = 1./Nprime *ivar *(Nprime*dxhat - np.sum(dxhat, axis=0) - xhat*np.sum(dxhat*xhat, axis=0))
    dx = np.reshape(dx.T, (N,C,H,W))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
