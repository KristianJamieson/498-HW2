"""This module contains functions to fit classification models for HW 2 in AE 498 Computational Systems Engineering.

"""

import numpy as np


def fit_knn(X, y, n_neighbors):
    """Function to fit a KNN classifier to a given dataset.

    Parameters
    ----------
    X : array_like
        Input data (n x (p + 1) dimensions), where each row is an observation and each column is a predictor.
    y : array_like
        Output/response data (n elements), where each element is an observation.
    n_neighbors : int
        Number of k neighbors to use for each prediction.

    Returns
    -------
    y_predicted : array_like
        Predicted responses for input data (n elements).
    error : float
        Error rate for the input data, calculated as 1/n_obs * number of incorrect predictions.

    Notes
    -----
    The test functions assume the k nearest neighbors of observation x include x.

    """

    n_obs = X.shape[0]
    y_predictions = []

    for i in range(n_obs):
        distances = np.sqrt(np.sum((X - X[i])**2, axis=1))

        nearest_indices = np.argsort(distances)[:n_neighbors]
        nearest_classes = y[nearest_indices]

        classes, counts = np.unique(nearest_classes, return_counts=True)
        predicted_class = classes[np.argmax(counts)]

        y_predictions.append(predicted_class)

    error = np.mean(np.array(y_predictions) != y)

    return error, y_predictions