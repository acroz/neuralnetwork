
import numpy as np
from matplotlib import pyplot
from neuralnetwork import NeuralNetwork
from neuralnetwork.gradientdescent import gradient_descent
import mnist

TRAINING_SET_SIZE = 50000
CROSSVAL_SET_SIZE = 10000
TESTING_SET_SIZE  = 10000

#TRAINING_SET_SIZE = 5000
#CROSSVAL_SET_SIZE = 1000
#TESTING_SET_SIZE  = 1000

def load(image_file, label_file):
    """
    Load and randomly shuffle a data set.
    """
    images = mnist.load_images(image_file)
    labels = mnist.load_labels(label_file)
    ishuffle = np.arange(images.shape[0])
    np.random.shuffle(ishuffle)
    return images[ishuffle,:,:], labels[ishuffle]

# Load data
main_images, main_labels = load(mnist.TRAIN_IMAGES, mnist.TRAIN_LABELS)
test_images, test_labels = load(mnist.TEST_IMAGES, mnist.TEST_LABELS)

# Determine the number of features (= number of voxels in image)
nfeatures = main_images.shape[1] * main_images.shape[2]

# Get training labels and features
train_images = main_images[:TRAINING_SET_SIZE,:,:]
train_labels = main_labels[:TRAINING_SET_SIZE]
train_features = train_images.reshape(train_images.shape[0], nfeatures).T

# Get cross validation labels and features
crsval_images = main_images[-CROSSVAL_SET_SIZE:,:,:]
crsval_labels = main_labels[-CROSSVAL_SET_SIZE:]
crsval_features = crsval_images.reshape(crsval_images.shape[0], nfeatures).T

# Get test validation labels and features
test_images = test_images[:TESTING_SET_SIZE,:,:]
test_labels = test_labels[:TESTING_SET_SIZE]
test_features = test_images.reshape(test_images.shape[0], nfeatures).T

# Get a sorted list of categories/labels
categories = sorted(np.unique(train_labels))

def convergence_rate_plot(network, optimisers, names, outfile=None):
    """
    Train the network with different optimisers and plot the convergence.
    """
    
    # Prepare the figure
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    
    for opt, lab in zip(optimisers, names):

        print('- Training with method', lab)

        # Reinitialise parameters with fixed random seed
        np.random.seed(5398299)
        network.initialize_parameters()

        # Train the network with lambda 1
        conv = network.train(train_features, train_labels, lmbda=1.,
                             optimiser=opt, maxiter=50, retconv=True)
        
        # Plot the convergence
        ax.plot(conv, label=lab)
    
    ax.set_xlabel('Iteration Count')
    ax.set_ylabel('Cost')
    pyplot.legend()
   
    if outfile is not None:
        pyplot.savefig(outfile)

def convergence_gradient_descent(outfile=None):

    print('Analysing convergence of gradient descent with different alphas')

    design = [nfeatures, 300, len(categories)]
    print('    Network shape: {}'.format(design))

    network = NeuralNetwork(design, categories)

    alphas = [0.1, 0.2, 0.4, 0.6, 1.0]
    optimisers = [gradient_descent(a) for a in alphas]
    names = ['$\\alpha={}$'.format(a) for a in alphas]
    convergence_rate_plot(network, optimisers, names, outfile)

def convergence_methods(outfile=None):

    print('Analysing convergence of gradient descent with different alphas')

    design = [nfeatures, 300, len(categories)]
    print('    Network shape: {}'.format(design))

    network = NeuralNetwork(design, categories)
    
    optimisers = [gradient_descent(0.4), 'CG', 'Newton-CG']
    names = ['Gradient Descent ($\\alpha=0.4$)', 'Conjugate Gradient',
             'Newton-CG']
    convergence_rate_plot(network, optimisers, names, outfile)

def accuracy_evolution(outfile=None):

    print('Plotting evolution of accuracy over time')

    design = [nfeatures, 300, len(categories)]
    print('    Network shape: {}'.format(design))

    network = NeuralNetwork(design, categories)

    conv = network.train(train_features, train_labels, lmbda=5., maxiter=200,
                         retaccur=(crsval_features, crsval_labels))

    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)

    ax.plot(conv[:,0], label='Training Accuracy')
    ax.plot(conv[:,1], label='CV Accuracy')

    ax.set_xlabel('Iteration Count')
    ax.set_ylabel('Accuracy')
    pyplot.legend(loc='lower right')
    
    if outfile is not None:
        pyplot.savefig(outfile)

def accuracy(network, lmbda=5.):
    
    print('- Training network shape {} with lambda {}'.format(network, lmbda))

    np.random.seed(342375145)
    network.initialize_parameters()
    network.train(train_features, train_labels, lmbda=lmbda, maxiter=100)

    train_accuracy = network.accuracy(train_features, train_labels)
    crossval_accuracy = network.accuracy(crsval_features, crsval_labels)

    return train_accuracy, crossval_accuracy

def trial_1layer_designs(outfile=None):

    hidden_layer_sizes = (np.arange(7) + 13) * 100 + 100

    train_accur = []
    cv_accur    = []

    for size in hidden_layer_sizes:

        design = [nfeatures, size, len(categories)]

        network = NeuralNetwork(design, categories)
        train, cv = accuracy(network, 0)

        train_accur.append(train)
        cv_accur.append(cv)

        with open('trial_1layer_designs_it200.log', 'a') as fp:
            fp.write('{} {} {}\n'.format(size, train, cv))

    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)

    ax.plot(hidden_layer_sizes, train_accur, 'x-', label='Training Set')
    ax.plot(hidden_layer_sizes, cv_accur, 'x-', label='Cross Validation Set')

    ax.set_xlabel('Hidden Layer Size')
    ax.set_ylabel('Cross Validation Accuracy')

    pyplot.legend()

    if outfile is not None:
        pyplot.savefig(outfile)

def trial_2layer_designs(outfile=None):

    first_layer_sizes = [500, 800, 1000]
    second_layer_sizes = [10, 50, 100, 500]

    results = []

    for first in first_layer_sizes:

        results.append([])

        for second in second_layer_sizes[2:]:
            
            if second is None:
                design = [nfeatures, first, len(categories)]
            else:
                design = [nfeatures, first, second, len(categories)]
            
            network = NeuralNetwork(design, categories)
            train_accur, cv_accur = accuracy(network, 0)

            with open('trial_2layer_designs_it100.log', 'a') as fp:
                fp.write('{} {} {} {}\n'.format(first, second, train_accur,
                                                cv_accur))

            results[-1].append(cv_accur)

    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)

    heat = ax.pcolor(np.array(results).T, cmap='Greens')

    ax.set_xlabel('First Layer Size')
    ax.set_ylabel('Second Layer Size')

    ax.set_xticks(np.arange(len(first_layer_sizes)) + 0.5)
    ax.set_yticks(np.arange(len(second_layer_sizes)) + 0.5)

    ax.set_xticklabels([str(v) for v in first_layer_sizes])
    ax.set_yticklabels([str(v) for v in second_layer_sizes])

    fig.colorbar(heat)

    if outfile is not None:
        pyplot.savefig(outfile)

def regularisation(outfile=None):

    print('Assessing best lambda to use')

    design = [nfeatures, 1000, len(categories)]
    network = NeuralNetwork(design, categories)

    lmbda_vals = [0, 0.1, 0.3, 1, 3, 10]
    train_accuracy = []
    crossval_accuracy = []

    for lmbda in lmbda_vals:
        ta, cva = accuracy(network, lmbda)
        train_accuracy.append(ta) 
        crossval_accuracy.append(cva)

        with open('regularisation_1000_it100.log', 'a') as fp:
            fp.write('{} {} {}\n'.format(lmbda, ta, cva))
    
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)

    ax.plot(lmbda_vals, train_accuracy, 'x-', label='Training')
    ax.plot(lmbda_vals, crossval_accuracy, 'x-', label='Cross Validation')
    
    ax.set_xlabel('Regularisation Parameter $\\lambda$')
    ax.set_ylabel('Accuracy')

    pyplot.legend()

    if outfile is not None:
        pyplot.savefig(outfile)

#accuracy_evolution('accuracy_evolution_full.svg')
#convergence_gradient_descent('gradient_descent_full.svg')
#convergence_methods('compare_optimisers_full.svg')
#trial_1layer_designs('design_1layer_full.svg')
trial_2layer_designs('design_2layer_full.svg')
#regularisation('regularisation.svg')

#pyplot.show()
