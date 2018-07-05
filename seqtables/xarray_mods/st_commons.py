import warnings
import numpy as np


def unique_variable(name, variables, compat='broadcast_equals'):
    # type: (Any, List[Variable], str) -> Variable
    """Return the unique variable from a list of variables or raise MergeError.

    Parameters
    ----------
    name : hashable
        Name for this variable.
    variables : list of xarray.Variable
        List of Variable objects, all of which go by the same name in different
        inputs.
    compat : {'identical', 'equals', 'broadcast_equals',
              'no_conflicts'}, optional
        Type of equality check to use.

    Returns
    -------
    Variable to use in the result.

    Raises
    ------
    MergeError: if any of the variables are not equal.
    """

    out = variables[0]

    for v in variables[1:]:
        assert v.shape == out.shape

    print(len(variables))
    if len(variables) > 1:
        return np.concatenate(variables)
    else:
        return out

    if len(variables) > 1:
        combine_method = None

        if compat == 'minimal':
            compat = 'broadcast_equals'

        if compat == 'broadcast_equals':
            dim_lengths = broadcast_dimension_size(variables)
            out = out.set_dims(dim_lengths)

        if compat == 'no_conflicts':
            combine_method = 'fillna'

        for var in variables[1:]:
            if False:  #not getattr(out, compat)(var):
                raise MergeError('conflicting values for variable %r on '
                                 'objects to be combined:\n'
                                 'first value: %r\nsecond value: %r'
                                 % (name, out, var))
            if combine_method:
                # TODO: add preservation of attrs into fillna
                out = getattr(out, combine_method)(var)
                out.attrs = var.attrs

    return out



def _maybe_promote_st(dtype):
    """
        Modified version of _maybe_promote found from xarray. This allows for ability to provide null values to
        ints and 'S1' datatype
    """
    # print('boobga')
    # N.B. these casting rules should match pandas
    if np.issubdtype(dtype, float):
        fill_value = np.nan
    elif np.issubdtype(dtype, int):
        # dtype = int
        fill_value = 0
    elif np.issubdtype(dtype, complex):
        fill_value = np.nan + np.nan * 1j
    elif np.issubdtype(dtype, np.datetime64):
        fill_value = np.datetime64('NaT')
    elif np.issubdtype(dtype, np.timedelta64):
        fill_value = np.timedelta64('NaT')
    elif np.issubdtype(dtype, 'S'):
        fill_value = ''  # fill with empty strings
    else:
        warnings.warning('CHECK THIS DATATYPE: ' + str(dtype))
        dtype = object
        fill_value = np.nan
    return np.dtype(dtype), fill_value
