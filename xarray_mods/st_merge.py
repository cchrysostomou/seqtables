import imp  # Standard module to do such things you want to.
from .st_commons import _maybe_promote_st, unique_variable
import xarray as xr
# We can import any module including standard ones:...this makes sure they are different copies of the normal xarray module
xr2 = imp.load_module('xr2', *imp.find_module('xarray'))
# overwrite merge method in xr so that we can allow empty strings as '' rather than np.nan and avoid dtype=Object
xr2.core.alignment._maybe_promote = _maybe_promote_st
xr2.core.merge.unique_variable = unique_variable


def merge(objects, compat='no_conflicts', join='outer'):
    # HACKILY convert object datatype to xr2
    hacky_types = [str(type(o)) for o in objects]

    conversions = [
        'Dataset' if 'core.dataset.Dataset' in ht else 'DataArray' if 'core.dataarray.DataArray' in ht else None
        for ht in hacky_types
    ]

    new_objects = [getattr(xr2, c)(objects[i]) if c is not None else objects[i] for i, c in enumerate(conversions)]
    merged_result = xr2.merge(new_objects, compat, join)
    new_type = str(type(merged_result))

    if 'core.dataset.Dataset' in new_type:
        merged_result = xr.Dataset(merged_result)
    elif 'core.dataarray.DataArray' in new_type:
        merged_result = xr.DataArray(merged_result)

    return merged_result


