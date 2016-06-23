"""
Retrieve sample data from the MNIST database.

http://yann.lecun.com/exdb/mnist/
"""

import gzip
import struct
import argparse
import numpy as np
from matplotlib import pyplot, gridspec
import random
from itertools import cycle

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
    """
    Download the MNIST data set.
    """
    print('Downloading files:')
    for filename in FILES:
        print('    {}'.format(filename))
        request.urlretrieve(MNIST_URL_ROOT + filename, filename)

def read_int32(fp):
    """
    Read a 32 bit integer from a file object.
    """
    return struct.unpack('>i', fp.read(4))[0]

def load_images(filename):
    """
    Load the images from an MNIST data file.
    """

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
    """
    Load the labels from an MNIST data file.
    """

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
    """
    Display a sample of images.
    """
   
    num_images = imagedata.shape[0]
    selection_size = nrows * ncols

    if num_images <= selection_size:
        selection = cycle(range(num_images))
    else:
        sel = []
        while len(sel) < selection_size:
            candidate = random.randint(0, num_images-1)
            if candidate not in sel:
                sel.append(candidate)
        selection = iter(sel)

    # Prepare a subplot grid
    gs = gridspec.GridSpec(nrows, ncols)
    gs.update(wspace=0.05, hspace=0.05)
    
    # Plot images in grayscale
    cmap = pyplot.get_cmap('gray')

    for irow in range(nrows):
        for icol in range(ncols):
            
            # Get the index of the image to display
            iimage = next(selection)

            # Create a subplot
            ax = pyplot.subplot(gs[irow, icol])

            # Plot the image
            ax.imshow(imagedata[iimage], cmap=cmap)

            # Turn off axis markers
            ax.set_xticks([])
            ax.set_yticks([])

            # Write label
            if labels is not None:
                pyplot.text(0.95, 0.03,
                            str(labels[iimage]),
                            fontsize=14, color='white',
                            horizontalalignment='right',
                            verticalalignment='bottom',
                            transform=ax.transAxes)

    pyplot.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--download',
                        action='store_true',
                        help='download the data files')
    
    parser.add_argument('--plot',
                        nargs='?', const=TRAIN_IMAGES,
                        help='plot a random sample of character images from '
                             'the specified file')

    parser.add_argument('--labels',
                        nargs='?', const=TRAIN_LABELS,
                        help='when plotting, draw these labels on the plot')

    args = parser.parse_args()

    if not args.download and args.plot is None:
        parser.error('no action specified - add --download and/or --plot')

    if args.download:
        download()

    if args.plot:
        image_data = load_images(args.plot)
        labels = None if args.labels is None else load_labels(args.labels)
        display(image_data, labels)

