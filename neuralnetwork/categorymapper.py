"""
Defines a mapper class for convenient mapping of categories and activations on
to each other.
"""

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
