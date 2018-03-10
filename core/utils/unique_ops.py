import pandas as pd
import numpy as np


def pandas_value_counts(df):
    """
    Simply apply the value_counts function to every column in a dataframe
    """
    return df.apply(pd.value_counts).fillna(0)


def numpy_value_counts_bin_count(arr, weights=None):
    """
    Use the 'bin count' function in numpy to calculate the unique values in every column of a dataframe
    clocked at about 3-4x faster than pandas_value_counts (df.apply(pd.value_counts))

    Args:
        arr (dataframe, or np array): Should represent rows as sequences and columns as positions. All values should be int
        weights (np array): Should be a list of weights to place on each
    """
    if isinstance(arr, pd.DataFrame):
        arr = arr.values
    elif not isinstance(arr, np.ndarray):
        raise Exception('The provided parameter for arr is not a dataframe or numpy array')
    if len(arr.shape) == 1:
        # its a ONE D array, lets make it two D
        arr = arr.reshape(-1, 1)

    bins = [np.bincount(arr[:, x], weights=weights) for x in range(arr.shape[1])]  # returns an array of length equal to the the max value in array + 1. each element represents number of times an integer appeared in array.
    indices = [np.nonzero(x)[0] for x in bins]  # only look at non zero bins
    series = [pd.Series(y[x], index=x) for (x, y) in zip(indices, bins)]
    return pd.concat(series, axis=1).fillna(0)


def custom_numpy_count(df, weights=None):
    """
    count all unique members in a numpy array and then using unique values, count occurrences at each position
    This is by far the slowest method (seconds rather than tenths of seconds)
    """
    val = df.values
    un = np.unique(val.reshape(-1))
    if weights:
        pass
    r = {u: np.einsum('i, ij->j', weights, (val == u)) if weights is not None else np.einsum('ij->j', (val == u).astype(int)) for u in un}
    return pd.DataFrame(r).transpose()
