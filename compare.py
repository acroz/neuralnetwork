
import numpy as np
from matplotlib import pyplot
from neuralnetwork import NeuralNetwork, mnist
from scipy.optimize import OptimizeResult

from book import Network as BookNet

def gradient_descent(alpha):
    """
    Generate a simple gradient descent optimiser.
    """

    def gradient_descent(fun, x0, args=(), jac=None, gtol=1e-5,
                         callback=None, maxiter=None, **kwargs):

        x = x0.copy()
        grad = jac(x)
        i = 0
        warnflag = 0
    
        while np.linalg.norm(grad) > gtol:
            
            i += 1
    
            grad = jac(x)
            x = x - alpha * grad
    
            if callback is not None:
               callback(x)
    
            if maxiter is not None and i >= maxiter:
                warnflag = 2
                break
    
        result = OptimizeResult(fun=fun(x), nit=i, nfev=1, njev=i,
                                status=warnflag, success=(warnflag==0), x=x)
    
        return result
    
    return gradient_descent

training_set_size = 10000
crossval_set_size = 1000
testing_set_size = 1000

# Load data
images = mnist.load_images(mnist.TRAIN_IMAGES)
labels = mnist.load_labels(mnist.TRAIN_LABELS)

# Randomly shuffle
ishuffle = np.arange(images.shape[0])
np.random.shuffle(ishuffle)
images = images[ishuffle,:,:]
labels = labels[ishuffle]

# Load test set
test_images = mnist.load_images(mnist.TEST_IMAGES)
test_labels = mnist.load_labels(mnist.TEST_LABELS)

# Randomly shuffle
ishuffle = np.arange(test_images.shape[0])
np.random.shuffle(ishuffle)
test_images = test_images[ishuffle,:,:]
test_labels = test_labels[ishuffle]

# Slice data
nfeatures = images.shape[1] * images.shape[2]

train_images = images[:training_set_size,:,:]
train_labels = labels[:training_set_size]
train_features = train_images.reshape(train_images.shape[0], nfeatures).T

crsval_images = images[-crossval_set_size:,:,:]
crsval_labels = labels[-crossval_set_size:]
crsval_features = crsval_images.reshape(crsval_images.shape[0], nfeatures).T

test_images = test_images[:testing_set_size,:,:]
test_labels = test_labels[:testing_set_size]
test_features = test_images.reshape(test_images.shape[0], nfeatures).T

import mnist_loader
btrain, bval, btest = mnist_loader.load_data_wrapper()

# Rearrange to vis
#btrainimg = np.array([e[0].reshape(28, 28) for e in btrain])
#btrainlab = np.array([e[1].argmax() for e in btrain])
#mnist.display(train_images, train_labels)
#mnist.display(btrainimg, btrainlab)
mnist.display(test_images, test_labels)

btrainmine = []
for img, lab in zip(train_features.T, train_labels):
    btrainmine.append((img.reshape(784,1), np.arange(10).reshape(10,1) == lab))
btestmine = []
for img, lab in zip(test_features.T, test_labels):
    btestmine.append((img.reshape(784,1), lab))

mynet = NeuralNetwork([784, 30, 10])
booknet = BookNet([784, 30, 10])

#booknet.SGD(btrain, 30, 10, 3.0, test_data=btest)
booknet.SGD(btrainmine, 30, 10, 3.0, test_data=btestmine)
