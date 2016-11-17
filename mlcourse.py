
import os
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize

from neuralnetwork.network import NeuralNetwork

# Theta == weights
# -threshold of neuron == bias
# i.e. if w = weights, x = inputs, b = bias, a = outputs:
#     a = bool(w.x + b > 0) # for "perceptron"
# Output of a sigmoid neuron is then a = sigmoid(w.x + b)
# sigmoid is the 'activation functon'
# z = w.x+b is the "weighted input"

ML_ROOT = '/home/acroz/coursera/machine_learning/ex4_neurallearning/ex4'

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def nnCostFunction(nn_params, input_layer_size, hidder_layer_size, num_labels,
                   X, y, lmda):
    #print("Entering cost function")
    
    # Recover weight arrays
    nt1 = hidder_layer_size * (input_layer_size + 1)
    Theta1 = nn_params[:nt1].reshape(hidder_layer_size, input_layer_size+1)
    Theta2 = nn_params[nt1:].reshape(num_labels, hidder_layer_size+1)

    # Setup useful vars
    m = X.shape[0]
    
    # Need to return following
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # Add ones to X data matrix
    a1 = np.hstack((np.ones((m, 1)), X))
    # Compute inner layer
    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2)

    # Add ones to inner layer
    a2 = np.hstack((np.ones((a2.shape[0], 1)), a2))
    # Compute output later
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)

    # Compute cost
    labels = np.arange(num_labels) + 1
    match = y[:,np.newaxis] == labels
    components = -(match * np.log(a3)) - (1 - match) * np.log(1 - a3)
    J = components.sum() / m

    # Regularised cost
    Theta1_reduced = Theta1.copy()
    Theta2_reduced = Theta2.copy()
    Theta1_reduced[:,0] = 0
    Theta2_reduced[:,0] = 0
    squared = (Theta1_reduced**2).sum() + (Theta2_reduced**2).sum()
    J += lmda/(2*m) * squared

    # Backpropagation
    # Compute error terms
    #d3 = np.zeros(a3.shape)
    #for i in range(num_labels):
    #    lab = i + 1
    #    d3[:,i] = a3[:,i] - (y == lab)
    d3 = a3 - match
    d2 = np.dot(d3, Theta2[:,1:]) * sigmoid_gradient(z2)
    
    #print("a3", a3[0,:])
    #print("d3", d3[0,:10])
    #print("d2", d2[0,:10])
    
    Theta1_grad = np.dot(d2.T, a1)/m + lmda/m * Theta1_reduced
    Theta2_grad = np.dot(d3.T, a2)/m + lmda/m * Theta2_reduced

    # Unroll
    grad = np.concatenate((Theta1_grad.ravel(), Theta2_grad.ravel()))

    #print("Leaving cost function, value:", J)

    return J, grad

def predict(Theta1, Theta2, X):
    m = X.shape[0]

    # Add ones to X data matrix
    a1 = np.hstack((np.ones((m, 1)), X))
    # Compute inner layer
    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2)

    # Add ones to inner layer
    a2 = np.hstack((np.ones((a2.shape[0], 1)), a2))
    # Compute output later
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)

    return a3.argmax(1) + 1

def sine_initialiser(shape):
    array = np.zeros(shape)
    view = array.ravel()
    view[:] = np.sin(np.arange(view.size))
    return array

def sin_init(fan_out, fan_in):
    nelem = fan_out * (1+fan_in)
    return np.sin(np.arange(nelem)).reshape(fan_out, fan_in+1)

def numerical_gradients(J, theta):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for i in range(theta.size):
        perturb[i] = e
        numgrad[i] = (J(theta + perturb)[0] - J(theta - perturb)[0]) / (2*e)
        perturb[i] = 0.
    return numgrad

def validate_gradients(lmda=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    Theta1 = sin_init(hidden_layer_size, input_layer_size)
    Theta2 = sin_init(num_labels, hidden_layer_size)

    X = sin_init(m, input_layer_size - 1)
    y = 1 + np.mod(np.arange(m)+1, num_labels)

    nn_params = np.concatenate((Theta1.ravel(), Theta2.ravel()))

    J = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size,
                                 num_labels, X, y, lmda)

    _,grad = J(nn_params)
    numgrad = numerical_gradients(J, nn_params)

    err = np.linalg.norm(grad - numgrad) / np.linalg.norm(grad + numgrad)
    return err

def numerical_gradients_class(J, theta):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for i in range(theta.size):
        perturb[i] = e
        numgrad[i] = (J(theta + perturb) - J(theta - perturb)) / (2*e)
        perturb[i] = 0.
    return numgrad

def validate_gradients_class(lmda=0):
   
    n_samples = m = 10
    n_features = 3
    n_hidden = 6
    n_outputs = 12

    net = NeuralNetwork([n_features, n_hidden, n_outputs])
    net.initialize_parameters(sine_initialiser)
    params = net.parameters

    X = sine_initialiser((n_features, m))
    y = np.mod(np.arange(n_samples)+1, n_outputs)
    proper_y = (y[:,np.newaxis] == np.arange(n_outputs)).T
    
    def J(p, gradient=False):
        net.parameters = p
        return net.cost(X, proper_y, lmda, gradient=gradient)

    _,grad = J(params, gradient=True)
    numgrad = numerical_gradients_class(J, params)

    err = np.linalg.norm(grad - numgrad) / np.linalg.norm(grad + numgrad)
    err = np.linalg.norm(grad - numgrad)
    print(np.linalg.norm(grad), np.linalg.norm(numgrad), np.linalg.norm(grad - numgrad))
    return err

net = NeuralNetwork([400, 25, 10])

weights = loadmat(os.path.join(ML_ROOT, 'ex4weights.mat'))

# keys Theta1, Theta2
Theta1 = np.array(weights['Theta1'])
Theta2 = np.array(weights['Theta2'])

data    = loadmat(os.path.join(ML_ROOT, 'ex4data1.mat'))
# keys X, y
X = np.array(data['X'])
y = np.array(data['y'])[:,0]

proper_y = y[:,np.newaxis] == (np.arange(10) + 1)

# Swap out 10-labels for 0-labels
#y[y==10] = 0
#Theta2 = np.roll(Theta2, -1, axis=0)

# Unroll params
#print(Theta1[:,0])
#Theta1_class = np.roll(Theta1, -1, axis=1)
#Theta2_class = np.roll(Theta2, -1, axis=1)
#print(Theta1_class[:,0])

bias1   = Theta1[:,0]
weight1 = Theta1[:,1:]
bias2   = Theta2[:,0]
weight2 = Theta2[:,1:]

nn_params = np.concatenate((Theta1.ravel(), Theta2.ravel()))
nn_params_class = np.concatenate((weight1.ravel(), bias1.ravel(),
                                  weight2.ravel(), bias2.ravel()))

net.parameters = nn_params_class

# Test cost function
print("\ncourse no reg")
J,_ = nnCostFunction(nn_params, 400, 25, 10, X, y, 0)
print('Cost func w/o reg = {} (ref: 0.287629)'.format(J))

print("\ncourse with reg")
J,_ = nnCostFunction(nn_params, 400, 25, 10, X, y, 1)
print('Cost func w/ reg  = {} (ref: 0.383770)'.format(J))

print("\nclass no reg")
Jwo,_ = net.cost(X.T, proper_y.T, gradient=True)
print("\nclass with reg")
Jw,_  = net.cost(X.T, proper_y.T, 1, gradient=True)
print('Class costs: w/o reg = {}, w/ reg = {}'.format(Jwo, Jw))

# Numerical gradient verification
if True:
    # Testing gradient function
    err = validate_gradients()
    print('Verify grad w/o reg: {} (should be < 1e-9)'.format(err))
    err = validate_gradients(3)
    print('Verify grad w/ reg:  {} (should be < 1e-9)'.format(err))
    
    # Testing gradient function
    err = validate_gradients_class()
    print('Verify class grad w/o reg: {} (should be < 1e-9)'.format(err))
    err = validate_gradients_class(3)
    print('Verify class grad w/ reg:  {} (should be < 1e-9)'.format(err))


# Train clas
net.train(X.T, y, 1)


1/0

# Initialise weights
epsilon = 0.12
init_Theta1 = np.random.rand(25, 401) * 2 * epsilon - epsilon
init_Theta2 = np.random.rand(10, 26)  * 2 * epsilon - epsilon
init_nn_params = np.concatenate((init_Theta1.ravel(), init_Theta2.ravel()))

# Train
lmda = 1
cost = lambda p: nnCostFunction(p, 400, 25, 10, X, y, lmda)

result = minimize(cost, init_nn_params, method='CG', jac=True,
                  options={'disp': True, 'maxiter': 50})

fitted_param = result.x

Theta1 = fitted_param[:init_Theta1.size].reshape(init_Theta1.shape)
Theta2 = fitted_param[init_Theta1.size:].reshape(init_Theta2.shape)

predictions = predict(Theta1, Theta2, X)
print(predictions.shape)
correct = (predictions == y).sum()
print(correct, y.size, float(correct)/y.size)
