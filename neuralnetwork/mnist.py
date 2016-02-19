"""
Retrieve sample data from the MNIST database.

http://yann.lecun.com/exdb/mnist/
"""

import gzip
import struct
import numpy as np
from matplotlib import pyplot, gridspec

try:
    from urllib import request
except ImportError:
    # Python 2 compatability
    from urllib2 import Request as request

MNIST_URL_ROOT = 'http://yann.lecun.com/exdb/mnist/'

TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES  = 't10k-images-idx3-ubyte.gz'
TEST_LABELS  = 't10k-labels-idx1-ubyte.gz'

FILES = [TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS]

def download():
    print('Downloading files:')
    for filename in FILES:
        print('    {}'.format(filename))
        request.urlretrieve(MNIST_URL_ROOT + filename, filename)

def read_int32(fp):
    return struct.unpack('>i', fp.read(4))[0]

def load_images(filename):

    with gzip.open(filename, 'rb') as fp:

        # Read header
        magic   = read_int32(fp)
        nimages = read_int32(fp)
        nrows   = read_int32(fp)
        ncols   = read_int32(fp)

        # Load raw data
        byte_str = fp.read()

    # Convert to numpy array
    dtype = np.dtype('uint8').newbyteorder('>')
    data = np.frombuffer(byte_str, dtype=dtype)

    # Check read correctly
    nvals = nimages * nrows * ncols
    assert nvals == data.size, 'incorrect number of values read'

    # Reshape
    data.shape = (nimages, nrows, ncols)

    return data

def load_labels(filename):

    with gzip.open(filename, 'rb') as fp:

        # Read header
        magic   = read_int32(fp)
        nlabels = read_int32(fp)

        # Load raw data
        byte_str = fp.read()
    
    # Convert to numpy array
    dtype = np.dtype('uint8').newbyteorder('>')
    data = np.frombuffer(byte_str, dtype=dtype)

    # Check read correctly
    assert nlabels == data.size, 'incorrect number of values read'

    return data

def display(imagedata, labels=None, nrows=5, ncols=6):

    image = iter(imagedata)
    if labels is not None:
        labels = iter(labels)

    # Prepare a subplot grid
    gs  = gridspec.GridSpec(nrows, ncols)
    gs.update(wspace=0.05, hspace=0.05)
    
    # Plot images in grayscale
    cmap = pyplot.get_cmap('gray')

    for irow in range(nrows):
        for icol in range(ncols):

            # Create a subplot
            ax = pyplot.subplot(gs[irow, icol])

            # Plot the image
            ax.imshow(next(image), cmap=cmap)

            # Turn off axis markers
            ax.set_xticks([])
            ax.set_yticks([])

            # Write label
            if labels is not None:
                pyplot.text(0.95, 0.03, str(next(labels)),
                            fontsize=14, color='greenyellow',
                            horizontalalignment='right',
                            verticalalignment='bottom',
                            transform=ax.transAxes)

    pyplot.show()
