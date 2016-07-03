"""
Implementation of a neural network layer.
"""

import numpy as np
from scipy.special import expit as sigmoid

try:
    import numexpr

    def sigmoid(z):
        """
        The sigmoid function.
    
        The sigmoid function is defined as:
    
        .. math::
    
            \\sigma(z) = \\frac{1}{1 + e^{-z}} 
        """
        return numexpr.evaluate('1/(1+exp(-z))')
    
    def sigmoid_prime(z):
        """
        The first derivative of the sigmoid function.
    
        The first derivative can be computed by: 
    
        .. math::
    
            \\sigma^\\prime(z) = \\sigma(z) (1 - \\sigma(z))
        """
        sig = sigmoid(z)
        return numexpr.evaluate('sig * (1 - sig)')

except ImportError:
    # numexpr speedup not critical - fall back to scipy

    def sigmoid(z):
        """
        The sigmoid function.
    
        The sigmoid function is defined as:
    
        .. math::
    
            \\sigma(z) = \\frac{1}{1 + e^{-z}} 
        """
        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoid_prime(z):
        """
        The first derivative of the sigmoid function.
    
        The first derivative can be computed by: 
    
        .. math::
    
            \\sigma^\\prime(z) = \\sigma(z) (1 - \\sigma(z))
        """
        sig = sigmoid(z)
        return sig * (1.0 - sig)

class LayerResult(object):
    """
    The result of a forward computation through a neural network layer.

    Parameters
    ----------
    x : array_like
        The inputs to the layer
    z : array_like
        The augmented inputs to the layer
    a : array_like
        The activations of the layer
    """

    def __init__(self, x, z, a):
        self.x = x
        self.z = z
        self.a = a

class NeuralNetworkLayer(object):
    """
    A layer in an artificial neural network.

    Parameters
    ----------
    inputs : int
        The number of inputs to this neural network layer.
    neurons : int
        The number of neurons in this neural network layer.
    """

    def __init__(self, inputs, neurons):
        
        # Store the attributes of the layer
        self._inputs  = int(inputs)
        self._neurons = int(neurons)

        # Allocate an array for the parameters of this layer
        n_weights = self._neurons * self._inputs
        n_biases  = self._neurons
        self._params = np.zeros(n_weights + n_biases)

        # Create views on params for weights and biases
        self._weights = self._params[:n_weights].reshape(self._neurons,
                                                         self._inputs)
        self._biases = self._params[n_weights:]

    @property
    def parameters(self):
        """
        Get the combined parameters for this layer.
        """
        return self._params

    @parameters.setter
    def parameters(self, array):
        """
        Set the combined parameters for this later.
        """
        # Copy into existing array
        self._params[:] = array

    def regularisation_penalty(self):
        """
        Compute the unscaled regularisation penalty for this layer.

        The regularisation penalty is computed as the sum of squares of the
        weights of this layer.

        Returns
        -------
        float
            The unscaled regularisation penalty
        """
        return (self._weights**2).sum()

    def forward(self, x):
        """
        Propagate the provided inputs forward through this network layer.

        Apply the relation:

        .. math::

            a = \\sigma(w \\cdot x + b)

        to calculate the activations of this layer from the provided inputs
        :math:`x` and the layer's weights :math:`w` and biases :math:`b`.
        :math:`\\sigma` represents the sigmoid function.

        The inputs to this function are expected to be a 2 dimensional array,
        where the second dimension corresponds to an independent input example.
        The first dimension corresponds to the number of neurons in this layer.

        Parameters
        ----------
        x : array_like
            Shape (num_neurons, num_examples) array of inputs to this layer

        Returns
        -------
        LayerResult
            A simple container object holding the inputs (result.x), augmented
            inputs (result.z) and activations (results.a) for this computation
        """

        # Check inputs make sense
        assert x.ndim == 2 and x.shape[0] == self._inputs

        # Compute augmented inputs
        z = np.dot(self._weights, x) + self._biases[:,np.newaxis]

        # Compute activations
        a = sigmoid(z)
        
        # Return the result
        return LayerResult(x, z, a)

    def backward(self, delta, z_layerm1):
        """
        Backpropagate the errors of this layer to its input layer.

        The backpropation algorithm allows the estimation of the errors
        :math:`\\delta` of layer l of a neural network from layer l+1 by the
        relation:

        .. math::

            \\delta^l = (w^{l+1}\\cdot\\delta^{l+1}) \\circ \\sigma^\\prime(z^l)

        where :math:`w^{l+1}` are the weights of layer l+1,
        :math:`\\sigma^\\prime` is the first derivative of the sigmoid function
        and :math:`z^l` is the augmented inputs of layer l:

        .. math::

            z^l = w^l \\cdot x^l + b^l

        :math:`\\circ` represents the Hadamard (element-wise) product.

        The inputs to this function are expected to be 2 dimensional arrays,
        where the second dimension corresponds to the errors/augmented inputs
        for each of a set of training examples. The first dimension of these
        arrays corresponds to the number of neurons and number of inputs in this
        layer, respectively.

        Parameters
        ----------
        delta : array_like
            Shape (num_neurons, num_training_examples) array of the errors on
            the neurons in this layer
        z_layerm1 : array_like
            Shape (num_inputs, num_training_examples) array of the augmented
            inputs to the input layer

        Returns
        -------
        ndarray
            The errors of the neurons on the input layer
        """

        # Check inputs make sense
        assert delta.ndim     == 2 and delta.shape[0]     == self._neurons
        assert z_layerm1.ndim == 2 and z_layerm1.shape[0] == self._inputs
        
        # Compute the deltas of the input layer using its provided augmented
        # inputs and this layer's deltas
        weights_T = np.ascontiguousarray(self._weights.T)
        return np.dot(weights_T, delta) * sigmoid_prime(z_layerm1)

    def gradient(self, delta, a_input, lmbda=0.0):
        """
        Compute the gradient of the cost function wrt this layer's parameters.

        The derivative of the cost function :math:`g` with respect to the weight
        :math:`w_{jk}^l` (the weight on the jth neuron by the kth input in layer
        l) can be computed from the errors :math:`\\delta_j^l` and input
        activations to this layer :math:`a_k^{l-1}`:

        .. math::

            \\frac{\\partial g}{\\partial w_{jk}^l} = \\delta_j^l a_k^{l-1}

        The derivative of the cost function :math:`g` with respect to the bias
        :math:`b_j^l` (the bias on the jth neuron in layer l) is equal to the
        errors :math:`\\delta_j^l`:

        .. math::
        
            \\frac{\\partial g}{\\partial b_j^l} = \\delta_j^l

        This function takes the errors for this layer, delta, and the input
        activations, a_input, and computes the gradients according to the above
        equations. These are expected to be 2 dimensional arrays, where the
        second dimension corresponds to the errors/input activations for each of
        a set of training examples. The first dimension of these arrays
        corresponds to the number of neurons and number of inputs in this layer,
        respectively.

        The gradients of the cost function with respect to the parameters of
        this network layer are then estimated by averaging over all examples in
        the training set, plus an additional term for the regularisation
        parameter :math:`\\lambda`:

        .. math::
            
            \\frac{\\partial g}{\\partial w_{jk}^l}
                = \\frac{1}{m} \\sum_{i=1}^m \\left[
                    \\frac{\\partial g}{\\partial w_{jk}^l}
                  \\right]_i
                + \\frac{\\lambda}{m} w_{jk}^l

        .. math::
            
            \\frac{\\partial g}{\\partial b_j^l}
                = \\frac{1}{m} \\sum_{i=1}^m \\left[
                    \\frac{\\partial g}{\\partial b_j^l}
                  \\right]_i

        Parameters
        ----------
        delta : array_like
            Shape (num_neurons, num_training_examples) array of the errors on
            the neurons in this layer
        a_input : array_like
            Shape (num_inputs, num_training_examples) array of the inputs to
            this network layer when computing the above errors
        lmbda : float, optional
            The regularisation parameter, defaults to 0.

        Returns
        -------
        ndarray
            The gradients of the cost function with respect to the parameters of
            this layer
        """

        # Check inputs make sense
        assert delta.ndim == 2   and delta.shape[0]   == self._neurons
        assert a_input.ndim == 2 and a_input.shape[0] == self._inputs

        # Get the number of samples from the second axis of the array
        n_samples = a_input.shape[1]

        # Compute the gradient of the cost function wrt the weights, averaged
        # over all the given samples
        # Ensure contiguous for efficient dot
        a_input_T = np.ascontiguousarray(a_input.T)
        grad_weights = np.dot(delta, a_input_T) / n_samples

        # Add on the gradient of the regularisation term
        if lmbda > 0.0:
            grad_weights += (lmbda/n_samples) * self._weights

        # Compute the gradient of the cost function wrt the biases, averaged
        # over all the given samples
        grad_biases = np.sum(delta, axis=1) / n_samples

        # Return a 1D array of the gradients in the same order as the paramters
        # property
        return np.concatenate((grad_weights.ravel(), grad_biases.ravel()))
