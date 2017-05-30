import numpy as np


def cartesian(arrays, out=None):
    """
    Code provided by: http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


def str_permutation(num_pos, alphabet, prefix='', suffix=''):
    """
      Returns all permutations of a string whose length is {num_pos} and whose allowed letters are defined by alphabet
    """
    arrs = list(prefix) + [alphabet] * num_pos + list(suffix)
    return cartesian(arrs).view('S' + str(len(prefix + suffix) + num_pos)).squeeze()


def dna_permutation(num_bases=3, prefix='', suffix='', alphabet=None, ):
    """
      Return all possible permutations of a DNA sequence of length {num_bases}
    """
    if alphabet is None:
        alphabet = ['A', 'C', 'G', 'T']
    return str_permutation(num_bases, alphabet, prefix, suffix)


def aa_permutation(num_res=1, prefix='', suffix='', no_stop=False, alphabet=None, ):
    """
      Return all possible permutations of a DNA sequence of length {num_bases}
    """
    if alphabet is None:
        alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '*']
    if no_stop:
        alphabet = [a for a in alphabet if a != '*']
    return str_permutation(num_res, alphabet, prefix, suffix)


def create_central_four(target):
    target = target.upper()
    ntdprefix = target[:9]
    ctdprefix = target[13:]
    return [dna for dna in dna_permutation(4, ntdprefix, ctdprefix) if dna != target]


def create_modular_var(target, module_length=3, exclude=[9, 10]):
    target = target.upper()
    arr = np.vstack([dna_permutation(module_length, target[:shift], target[(shift + module_length):]) for shift in range(22 - module_length + 1) if shift not in exclude]).squeeze().reshape(-1)
    return list(arr[arr != target])
