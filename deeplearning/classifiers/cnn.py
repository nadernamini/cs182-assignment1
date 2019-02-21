import numpy as np

from deeplearning.layers import *
from deeplearning.fast_layers import *
from deeplearning.layer_utils import *


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
        # Initialize weights and biases for the three-layer convolutional          #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        H, W = input_dim[1:]
        self.params['W1'] = np.random.normal(scale=weight_scale,
                                             size=(num_filters, input_dim[0], filter_size, filter_size))
        self.params['b1'] = np.zeros((num_filters,))
        self.params['W2'] = np.random.normal(scale=weight_scale,
                                             size=(num_filters * ((H - 2) / 2 + 1) * ((W - 2) / 2 + 1), hidden_dim))
        self.params['b2'] = np.zeros((hidden_dim,))
        self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b3'] = np.zeros((num_classes,))
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
        # Implement the forward pass for the three-layer convolutional net,        #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        out, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out, cache2 = affine_relu_forward(out, W2, b2)
        scores, cache3 = affine_forward(out, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # Implement the backward pass for the three-layer convolutional net,       #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dx = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))

        dx, grads['W3'], grads['b3'] = affine_backward(dx, cache3)
        dx, grads['W2'], grads['b2'] = affine_relu_backward(dx, cache2)
        dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx, cache1)

        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2
        grads['W3'] += self.reg * W3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class MultiLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - sbatchnorm - relu - dropout - 2x2 max pool - affine - batchnorm - relu - dropout - affine - batchnorm - softmax

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
        # Initialize weights and biases for the three-layer convolutional          #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        H, W = input_dim[1:]
        self.params['W1'] = np.random.normal(scale=weight_scale,
                                             size=(num_filters, input_dim[0], filter_size, filter_size))
        self.params['b1'] = np.zeros((num_filters,))
        self.params['W2'] = np.random.normal(scale=weight_scale,
                                             size=(num_filters * ((H - 2) / 2 + 1) * ((W - 2) / 2 + 1), hidden_dim))
        self.params['b2'] = np.zeros((hidden_dim,))
        self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b3'] = np.zeros((num_classes,))
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
        # Implement the forward pass for the three-layer convolutional net,        #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        out, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out, cache2 = affine_relu_forward(out, W2, b2)
        scores, cache3 = affine_forward(out, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # Implement the backward pass for the three-layer convolutional net,       #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dx = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))

        dx, grads['W3'], grads['b3'] = affine_backward(dx, cache3)
        dx, grads['W2'], grads['b2'] = affine_relu_backward(dx, cache2)
        dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx, cache1)

        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2
        grads['W3'] += self.reg * W3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class ConvNet(object):
    """
    A multi-layer convolutional network with the following architecture:

    {conv - [sbatchn] - relu - conv - [sbatchn] - relu - 2x2 max pool - [dropout]} x N - {affine} x M - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, hidden_dims=(100,), dropouts=(0.2, 0.3, 0.4), num_filters=(32, 64, 128), N=1, M=3,
                 input_dim=(3, 32, 32), filter_size=3, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32, seed=None, use_batchnorm=False):
        """
        Initialize a new network.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - dropouts: A list of scalars between 0 and 1 giving dropout strength for
          each repeating unit of the first unit. If dropout=0 then the network
          should not use dropout at all.
        - num_filters: A list of number of filters to use in each convolutional
          layer.
        - N: An integer giving the number of repeating units for the first unit.
        - M: An integer giving the number of repeating units for the second unit.
        - input_dim: Tuple (C, H, W) giving size of input data
        - filter_size: Size of filters to use in the convolutional layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deterministic so we can gradient check the
          model.
        - use_batchnorm: Whether or not the network should use batch normalization.
        """
        self.params = {}
        self.dropout_param = {}
        self.bn_params = {}
        self.reg = reg
        self.dtype = dtype
        self.filter_size = filter_size
        self.use_batchnorm = use_batchnorm
        self.use_dropout = len(dropouts) > 0
        self.N = N
        self.M = M
        C, H, W = input_dim

        def find_size_after_mp(H, W, times=self.N):
            for _ in range(times):
                H, W = (H - 2) / 2 + 1, (W - 2) / 2 + 1
            return H * W

        ############################################################################
        # Initialize weights and biases for the three-layer convolutional          #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        #                                                                          #
        #                                                                          #
        # When using dropout we need to pass a dropout_param dictionary to each    #
        # dropout layer so that the layer knows the dropout probability and the    #
        # mode (train / test). You can pass the same dropout_param to each dropout #
        # layer.                                                                   #
        # With batch normalization we need to keep track of running means and      #
        # variances, so we need to pass a special bn_param object to each batch    #
        # normalization layer. You should pass self.bn_params[0] to the forward    #
        # pass of the first batch normalization layer, self.bn_params[1] to the    #
        # forward pass of the second batch normalization layer, etc.               #
        ############################################################################
        H, W = input_dim[1:]
        filter_nums = [C] + list(num_filters)
        for i in range(self.N):
            self.params['W%s_1' % i] = np.random.normal(scale=weight_scale,
                                                        size=(num_filters[i], filter_nums[i], self.filter_size,
                                                              self.filter_size))
            self.params['b%s_1' % i] = np.zeros((num_filters[i],))
            if self.use_batchnorm:
                self.params['gamma%s_1' % i] = np.ones((num_filters[i],))
                self.params['beta%s_1' % i] = np.zeros((num_filters[i],))
                self.bn_params['%s_1' % i] = {'mode': 'train'}
            self.params['W%s_2' % i] = np.random.normal(scale=weight_scale,
                                                        size=(num_filters[i], filter_nums[i + 1], self.filter_size,
                                                              self.filter_size))
            self.params['b%s_2' % i] = np.zeros((num_filters[i],))
            if self.use_batchnorm:
                self.params['gamma%s_2' % i] = np.ones((num_filters[i],))
                self.params['beta%s_2' % i] = np.zeros((num_filters[i],))
                self.bn_params['%s_2' % i] = {'mode': 'train'}
            if self.use_dropout:
                self.dropout_param[i] = {'mode': 'train', 'p': dropouts[i]}
                if seed is not None:
                    self.dropout_param[i]['seed'] = seed
        hidden_dims = [num_filters[-1] * find_size_after_mp(H, W)] + list(hidden_dims)
        for i in range(self.M):
            self.params['W%s' % (self.N + i)] = np.random.normal(scale=weight_scale,
                                                                 size=(hidden_dims[i], hidden_dims[i + 1]))
            self.params['b%s' % (self.N + i)] = np.zeros((hidden_dims[i + 1],))

        self.params['W%s' % (self.N + self.M)] = np.random.normal(scale=weight_scale,
                                                                  size=(hidden_dims[-1], num_classes))
        self.params['b%s' % (self.N + self.M)] = np.zeros((num_classes,))

        # Cast all parameters to the correct datatype
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        # pass conv_param to the forward pass for the convolutional layer
        conv_param = {'stride': 1, 'pad': (self.filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # Implement the forward pass for the three-layer convolutional net,        #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        out, caches, l2_reg = X, {}, 0.0
        for i in range(self.N):
            out, caches['c%s_1' % i] = conv_forward_fast(out, self.params['W%s_1' % i],
                                                         self.params['b%s_1' % i], conv_param)
            l2_reg += np.sum(self.params['W%s_1' % i] ** 2)
            if self.use_batchnorm:
                out, caches['b%s_1' % i] = spatial_batchnorm_forward(out, self.params['gamma%s_1' % i],
                                                                     self.params['beta%s_1' % i],
                                                                     self.bn_params['%s_1' % i])
            out, caches['r%s_1' % i] = relu_forward(out)
            out, caches['c%s_2' % i] = conv_forward_fast(out, self.params['W%s_2' % i],
                                                         self.params['b%s_2' % i], conv_param)
            l2_reg += np.sum(self.params['W%s_2' % i] ** 2)
            if self.use_batchnorm:
                out, caches['b%s_2' % i] = spatial_batchnorm_forward(out, self.params['gamma%s_2' % i],
                                                                     self.params['beta%s_2' % i],
                                                                     self.bn_params['%s_2' % i])
            out, caches['r%s_2' % i] = relu_forward(out)
            out, caches['mp%s' % i] = max_pool_forward_fast(out, pool_param)
            if self.use_dropout:
                out, caches['d%s' % i] = dropout_forward(out, self.dropout_param[i])
        for i in range(self.M):
            out, caches['a%s' % (self.N + i)] = affine_forward(out, self.params['W%s' % (self.N + i)],
                                                               self.params['b%s' % (self.N + i)])
            l2_reg += np.sum(self.params['W%s' % (self.N + i)] ** 2)

        scores, caches['a%s' % (self.N + self.M)] = affine_forward(out, self.params['W%s' % (self.N + self.M)],
                                                                   self.params['b%s' % (self.N + self.M)])
        l2_reg += np.sum(self.params['W%s' % (self.N + self.M)] ** 2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # Implement the backward pass for the three-layer convolutional net,       #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dx = softmax_loss(scores, y)
        loss += 0.5 * self.reg * l2_reg

        dx, grads['W%s' % (self.N + self.M)], grads['b%s' % (self.N + self.M)] = \
            affine_backward(dx, caches['a%s' % (self.N + self.M)])
        grads['W%s' % (self.N + self.M)] += self.reg * self.params['W%s' % (self.N + self.M)]
        for i in range(self.M - 1, -1, -1):
            dx, dw, db = affine_backward(dx, caches['a%s' % (self.N + i)])
            grads['W%s' % (self.N + i)], grads['b%s' % (self.N + i)] = dw + self.reg * self.params[
                'W%s' % (self.N + i)], db

        for i in range(self.N - 1, -1, -1):
            if self.use_dropout:
                dx = dropout_backward(dx, caches['d%s' % i])
            dx = max_pool_backward_fast(dx, caches['mp%s' % i])
            dx = relu_backward(dx, caches['r%s_2' % i])
            if self.use_batchnorm:
                dx, dgamma, dbeta = spatial_batchnorm_backward(dx, caches['b%s_2' % i])
                grads['gamma%s_2' % i], grads['beta%s_2' % i] = dgamma, dbeta
            dx, dw, db = conv_backward_fast(dx, caches['c%s_2' % i])
            grads['W%s_2' % i], grads['b%s_2' % i] = dw + self.reg * self.params['W%s_2' % i], db
            dx = relu_backward(dx, caches['r%s_1' % i])

            if self.use_batchnorm:
                dx, dgamma, dbeta = spatial_batchnorm_backward(dx, caches['b%s_1' % i])
                grads['gamma%s_1' % i], grads['beta%s_1' % i] = dgamma, dbeta
            dx, dw, db = conv_backward_fast(dx, caches['c%s_1' % i])
            grads['W%s_1' % i], grads['b%s_1' % i] = dw + self.reg * self.params['W%s_1' % i], db
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


pass
