import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


def affine_batch_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  b, batch_cache = batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(b)
  cache = (fc_cache, batch_cache, relu_cache)
  return out, cache


def affine_batch_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, batch_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  d_batch, d_gamma, d_beta = batchnorm_backward(da, batch_cache)
  dx, dw, db = affine_backward(d_batch, fc_cache)
  return dx, dw, db, d_gamma, d_beta


def conv_batch_relu_pool_forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  b, batch_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  s, relu_cache = relu_forward(b)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, batch_cache, relu_cache, pool_cache)
  return out, cache


def conv_batch_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, batch_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  db, d_gamma, d_beta = spatial_batchnorm_backward(da, batch_cache)
  dx, dw, db = conv_backward_fast(db, conv_cache)
  return dx, dw, db, d_gamma, d_beta


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
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
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim

    self.params['W1'] = np.random.normal(scale=weight_scale, size=(num_filters, C, filter_size, filter_size))
    self.params['b1'] = np.zeros(num_filters)

    stride = 1
    pad = (filter_size - 1) / 2

    H_ = (H - filter_size + 2 * pad) / stride + 1
    W_ = (W - filter_size + 2 * pad) / stride + 1
    K_ = num_filters

    pool_size = 2
    pool_stride = 2

    H__ = (H_ - pool_size) / pool_stride + 1
    W__ = (W_ - pool_size) / pool_stride + 1

    self.bn_param1 = {'mode': 'train'}
    self.params['gamma1'] = np.ones(num_filters)
    self.params['beta1'] = np.zeros(num_filters)

    self.params['W2'] = np.random.normal(scale=weight_scale, size=(K_ * H__ * W__, hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)

    self.bn_param2 = {'mode': 'train'}
    self.params['gamma2'] = np.ones(hidden_dim)
    self.params['beta2'] = np.zeros(hidden_dim)

    self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
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
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out_conv_relu_pool, cache_conv_relu_pool = conv_batch_relu_pool_forward(X, W1, b1, self.params['gamma1'],
                                                                            self.params['beta1'], conv_param,
                                                                            self.bn_param1, pool_param)
    out_affine_relu, cache_affine_relu = affine_batch_relu_forward(out_conv_relu_pool, W2, b2, self.params['gamma2'],
                                                                   self.params['beta2'], self.bn_param2)
    scores, cache_affine = affine_forward(out_affine_relu, W3, b3)
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
    ############################################################################
    loss, d_scores = softmax_loss(scores, y)

    d_affine, grads['W3'], grads['b3'] = affine_backward(d_scores, cache_affine)
    d_affine_relu, grads['W2'], grads['b2'], grads['gamma2'], grads['beta2'] = \
        affine_batch_relu_backward(d_affine, cache_affine_relu)
    _, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] = \
        conv_batch_relu_pool_backward(d_affine_relu, cache_conv_relu_pool)

    # noinspection PyTypeChecker
    loss += 0.5 * self.reg * sum(np.sum(self.params[param] ** 2) for param in self.params.keys()
                                 if param.startswith('W'))

    for param in self.params.keys():
      if param.startswith('W'):
        grads[param] += self.reg * self.params[param]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
