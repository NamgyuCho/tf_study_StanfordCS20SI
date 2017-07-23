"""Reference: amodi's GIST (https://gist.github.com/ambodi/408301bc5bc07bc5afa8748513ab9477)"""

import tensorflow as tf
import sys
import numpy as np
import collections
from tensorflow.python.framework import dtypes

class DataSet(object):
    """Dataset class object."""

    def __init__(self, images, labels):
        """Initialize the class."""
        self._images = images
        self._num_examples = images.shape[0]
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
            return self._images

    @property
    def labels(self):
            return self._labels

    @property
    def num_examples(self):
            return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def num_batches(self, batch_size):
        return int(self._num_examples / batch_size)

    def next_batch(self, batch_size):
        """Return the next 'batch_size' examples from this dataset."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Current epoch is finished
            self._epochs_completed += 1

            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def read_data_sets(filename):
    # Read file
    X = []
    Y = []
    first_line = True
    with open(filename) as file_obj:
        for line in file_obj:
            tmp = line.strip().split()

            if first_line:
                first_line = False
                continue

            if tmp[4] == '"Present"':
                tmp[4] = '1'
            else:
                tmp[4] = '0'

            tmp1 = [float(i) for i in tmp]
            len_line = len(tmp1)
            X.append(tmp1[:-2])
            Y.append(tmp1[-1])
    X = np.array(X)
    Y = np.array(Y)
    num_example = len(Y)
    Y = np.reshape(Y, (num_example, 1))

    """Set the images and labels."""
    num_training = int(num_example*0.45)
    num_validation = int(num_example*0.1)
    num_test = num_example - num_training - num_validation

    mask = range(num_training)
    train_X = X[mask]
    train_Y = Y[mask]

    mask = range(num_training, num_training + num_validation)
    validation_X = X[mask]
    validation_Y = Y[mask]

    mask = range(num_training + num_validation, num_training + num_validation + num_test)
    test_X = X[mask]
    test_Y = Y[mask]

    train = DataSet(train_X, train_Y)
    validation = DataSet(validation_X, validation_Y)
    test = DataSet(test_X, test_Y)
    ds = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

    return ds(train=train, validation=validation, test=test)


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot

