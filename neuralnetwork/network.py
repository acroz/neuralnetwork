
import numpy as np
from scipy.optimize import minimize
from neuralnetwork.layer import NeuralNetworkLayer


def uniform_initializer(shape, epsilon=0.12):
    """
    Initialise array to a uniform interval centred on zero.
    """
    return np.random.rand(*shape) * 2 * epsilon - epsilon


def gaussian_initializer(shape, sigma=1.0):
    """
    Initialise array with a Gaussian distribution centred on zero.
    """
    return np.random.randn(*shape) * sigma


class CategoryMapper(object):
    """
    Map between categories and output activations.

    Parameters
    ----------
    size : int
        The number of categories to map
    """

    def __init__(self, size):
        self._size = int(size)
        self._categories = None

    def register(self, labels):
        """
        Register some categories for a set of labels.

        Parameters
        ----------
        labels : array_like
            A set of labels to register categories for.
        """
        self._categories = np.unique(labels)
        assert len(self._categories) == self._size

    def activations(self, labels):
        """
        Map a set of labels to their corresponding activations.

        Parameters
        ----------
        labels : array_like
            The labels to map

        Returns
        -------
        activations : array
            The corresponding activations
        """
        assert self._categories is not None
        return labels == self._categories[:,np.newaxis]

    def labels(self, activations):
        """
        Map a set of activations to their corresponding labels.

        Parameters
        ----------
        activations : array_like
            The activations to map

        Returns
        -------
        labels : array
            The corresponding labels
        """
        assert self._categories is not None
        return self._categories[activations.argmax(axis=0)]


class NeuralNetwork(object):
    """
    An artificial neural network.

    Parameters
    ----------
    design : array_like
        The number of neurons in each layer of the network

    To construct and train a network with size 100 input layer, two size 50
    hidden layers, and a size 10 output layer:

    >>> net = NeuralNetwork([100, 50, 50, 10])
    >>> # Load features and labels
    >>> # ...
    >>> lmbda = 1.0
    >>> net.train(features, labels, lmbda)

    Then, to make predictions on a test set:

    >>> predictions = net.predict(test_features)
    """

    def __init__(self, design):
      
        # Assemble the layers
        self._layers = []
        for inputs, neurons in zip(design[:-1], design[1:]):
            self._layers.append(NeuralNetworkLayer(inputs, neurons))
        
        # Store some useful attributes about the network
        self._inputs  = design[0]
        self._outputs = design[-1]

        self._mapper = CategoryMapper(self._outputs)
    
    @property
    def parameters(self):
        """
        Get the combined parameters for the full neural network.
        """
        return np.concatenate([lyr.parameters for lyr in self._layers])

    @parameters.setter
    def parameters(self, array):
        """
        Set the combined parameters for the full neural network.
        """

        start = 0

        # Loop over all layers
        for layer in self._layers:

            # Get the number of parameters for this layer
            length = layer.parameters.size

            # Assign the correct slice
            layer.parameters = array[start:start+length]

            # Get the starting point of the next layer
            start += length

    def initialize_parameters(self, func=uniform_initializer):
        """
        Initialise the network's parameters using the provided function.

        Parameters
        ----------
        func : callable
            Callable returning an array_like of initial parameter values when
            given the number of parameters as a single argument
        """
        self.parameters = np.array(func(self.parameters.shape))

    def forward(self, input_activations):
        """
        Calculate the activations of the output layer of the network.

        Parameters
        ----------
        input_activations : array_like
            The activations of the input layer of the network. The first
            dimension should be equal in size to the input layer.

        Returns
        -------
        array
            The activations of the output layer.
        """

        # For compactness
        a = input_activations
        
        # Layers expect 2D arrays - this conversion enables support of
        # processing single input activation sets
        if a.ndim == 1:
            a = a[:,np.newaxis]
        
        # Check inputs
        assert a.ndim == 2 and a.shape[0] == self._inputs

        # Propagate through the network layers
        for layer in self._layers:
            a = layer.forward(a).a

        # Return the result
        return a

    def predict(self, features):
        """
        Predict the output category for a set of input activations.

        Parameters
        ----------
        features : array_like
            The activations of the input layer of the network. The first
            dimension should be equal in size to the input layer.

        Returns
        -------
        array
            The predicted categories.
        """
        
        # Compute activations
        activations = self.forward(features)

        # Recover original category labels
        return self._mapper.labels(activations)

    def accuracy(self, features, labels):
        """
        Compute the predictive accuracy of the network.

        Provided a set of input activations and their corrsponding category
        labels, this function computes the predictions of the neural network
        and returns the fraction that were correct.

        Parameters
        ----------
        features : array_like
            The activations of the input layer of the network. The first
            dimension should be equal in size to the input layer.
        labels : array_like
            The correct labels for the provided examples.

        Returns
        -------
        float
            The fraction of predictions that were correct.
        """
        return (self.predict(features) == labels).sum() / float(len(labels))
        
    def cost(self, features, outputs, lmbda=0.0, gradient=False):
        """
        Evaluate the cost function for the network from a set of examples.
        
        Parameters
        ----------
        features : array_like
            The features, or input activations - shape (nfeatures, nexamples)
        outputs : array_like
            The 'true' output activations - shape (noutput, nexamples)
        lmbda : float, optional
            The regularisation parameter to use (default: 0.0)
        gradient : bool, optional
            If True, also return the gradient of the cost function (default: 
            False)

        Returns
        -------
        float
            The cost
        array
            The gradient of the cost function, when `gradient` is True
        """
        
        # Sanity checks
        assert features.shape[0] == self._inputs
        assert outputs.shape[0]  == self._outputs
        assert features.shape[1] == outputs.shape[1]
        
        # Get once
        n_examples = features.shape[1]

        # Propagate activations through network, storing the results
        a = features
        results = []
        for layer in self._layers:
            res = layer.forward(a)
            results.append(res)
            a = res.a
        
        # Evaluate the cross entropy cost function
        components = -(outputs * np.log(a)) - (1 - outputs) * np.log(1 - a)
        cost = components.sum() / n_examples

        # Regularisation
        if lmbda > 0:
            scale = lmbda / (2 * n_examples)
            for layer in self._layers:
                cost += scale * layer.regularisation_penalty()
       
        # Don't compute gradient if not requested
        if not gradient:
            return cost
        
        # Use backwards propagation algorithm to compute gradients
        # Compute errors in final layer
        delta = a - outputs

        gradient_parts = []

        for layer, res, res_prev in zip(self._layers[::-1],
                                        results[::-1][:-1],
                                        results[::-1][1:]):
            
            # Compute and store gradients for this layer
            grad = layer.gradient(delta, res.x, lmbda)
            gradient_parts.append(grad.ravel())
            
            # Backpropagate errors
            delta = layer.backward(delta, res_prev.z)
        
        # Compute and store gradients for final layer
        grad = self._layers[0].gradient(delta, results[0].x, lmbda)
        gradient_parts.append(grad.ravel())

        # Stick gradients together in forward order
        gradient_flat = np.concatenate(gradient_parts[::-1])

        return cost, gradient_flat

    def cost_from_labels(self, features, labels, *args, **kwargs):
        """
        Convenience function to map labels to activations and compute the cost.        
        """
        activations = self._mapper.activations(labels)
        return self.cost(features, activations, *args, **kwargs)
    
    def train(self, features, labels, lmbda=0.0, optimiser='CG', retconv=False,
              retaccur=None, maxiter=None, **kwargs):
        """
        Train the neural network to a set of examples.
        
        Parameters
        ----------
        features : array_like
            The features, or input activations, of the training examples.
        labels : array_like
            The labels, or output categories, of the training examples.
        lmbda : float, optional
            The regularisation parameter (default: 0.0)
        optimiser : str, optional
            The optimisation method, passed to scipy.optimize.minimize (default:
            'CG')
        retconv : bool, optional
            Return the history of the convergence of the algorithm
        retaccur : bool, optional
            Return the history of the accuracy of the training set and a
            validation set
        """

        # Store a list of unique categories
        self._mapper.register(labels)

        # Evaluate a matrix of output activations for training
        activations = self._mapper.activations(labels)
        
        # Randomly initialise the parameters of the network
        self.initialize_parameters()

        # Cost function for optimiser
        def cost(params):
            self.parameters = params
            return self.cost(features, activations, lmbda, gradient=True)
        
        # Set up any relevant callback functions
        callback = None
        if retconv:
            convergence = []
            def callback(params):
                self.parameters = params
                convergence.append(self.cost(features, activations, lmbda))
        if retaccur is not None:
            convergence = []
            def callback(params):
                self.parameters = params
                convergence.append((self.accuracy(features, labels),
                                    self.accuracy(*retaccur)))
        if callback is not None:
            # Initial value
            callback(self.parameters)
        
        # Train the network
        result = minimize(cost, self.parameters, method=optimiser, jac=True,
                          callback=callback, options={'maxiter': maxiter})
        
        # Set network parameters to optimal values
        self.parameters = result.x
        
        if retconv or retaccur is not None:
            return np.array(convergence)
