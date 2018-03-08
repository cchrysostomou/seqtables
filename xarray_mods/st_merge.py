import imp  # Standard module to do such things you want to.
# from .st_commons import _maybe_promote_st, unique_variable
import xarray as xr
# We can import any module including standard ones:...this makes sure they are different copies of the normal xarray module
# xr2 = imp.load_module('xr2', *imp.find_module('xarray'))
# overwrite merge method in xr so that we can allow empty strings as '' rather than np.nan and avoid dtype=Object
# xr2.core.alignment._maybe_promote = _maybe_promote_st
# xr2.core.merge.unique_variable = unique_variable
import numpy as np
import pandas as pd
import warnings
import sys
from collections import OrderedDict

def merge_attributes(attributes, assert_unique_insertions):    
    # check version
    if sys.version_info[0] == 3 and sys.version_info[1] >= 5:
        # merge_fxn = lambda x, y: {**x, **y}
        merge_fxn = merge_two_dicts_py25
    else:
        merge_fxn = merge_two_dicts_py25
    
    ins_df = pd.concat([a.get('seqtable', {}).get('insertions', pd.DataFrame()) for a in attributes])
    if assert_unique_insertions:
        assert ins_df.index.drop_duplicates().shape[0] == ins_df.index.shape[0], 'Error we found duplicate rows when concatenating insertions'    
    
    new_attributes = OrderedDict({'seqtable': {}, 'user_defined': {}})
    
    references = []    
    
    for a in attributes:
        new_attributes['user_defined'] = merge_fxn(new_attributes['user_defined'], a.get('user_defined', {}))
        new_attributes['seqtable'] = merge_fxn(new_attributes['seqtable'], a.get('seqtable', {}))
        r = a.get('seqtable', {}).get('references')
        if r:            
            references.append(r)
    
    new_attributes['seqtable']['insertions'] = ins_df
    new_attributes['seqtable']['references'] = references
                
    return new_attributes
    
def merge_two_dicts_py25(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


def st_merge_arrays(objects, ignore_read_index=True, join='outer', assert_unique_insertions=True):
    """
        Create a very simple/primitive way to concat a set of seq table arrays. Using standard pandas or xarray is problematic 'out-of-the-box' because cannot treat treat empty cells as a string or value of choice. instead they are filled as NaN values.

        This will not check for conflicts (tables that have the same row-column positions. it will just overwrite them)
    """
    are_data_arrays = [isinstance(o, xr.DataArray) for o in objects]
    assert sum(are_data_arrays) == len(objects)
    
    # at least one array has quality
    quality_preset = sum([o.attrs.get('seqtable', {}).get('has_quality', False) for o in objects]) > 0
    
    # make sure certain attributes are constant
    seq_types = set([o.attrs.get('seqtable', {}).get('seq_type', None) for o in objects])
    seq_types.discard(None)
    assert len(seq_types) == 1, "The sequence types in the provided tables do not match"
           
    phred_adjust = set([o.attrs.get('seqtable', {}).get('phred_adjust', None) for o in objects])
    phred_adjust.discard(None)
    if len(seq_types) > 1:
        warnings.warn('Warning, the provided arrays have different options for phred values. Will only select one of the options for the merged table')
    
    fill_na = set([o.attrs.get('seqtable', {}).get('fillna_value', None) for o in objects])
    fill_na.discard(None)
    if len(fill_na) > 1:
        warnings.warn('Warning, the provided arrays have different options for fill_na vales. Will only select one of the options for the merged table')
               
    new_attributes = merge_attributes([o.attrs for o in objects], assert_unique_insertions)
    
    read_idx = []
    position_idx = []
    for o in objects:
        read_idx.extend(list(o.read.values))
        position_idx.extend(list(o.position.values))
    read_idx = set(read_idx) if ignore_read_index is True else list(read_idx)
    position_idx = set(position_idx)

    init_seq = np.empty((len(read_idx), len(position_idx)), dtype='S1')
    init_seq[:, :] = '-'  # this is a gap

    init_qual = np.empty((len(read_idx), len(position_idx)), dtype='S1')
    init_qual[:, :] = '!'  # lowest 
    
    new_xar = xr.DataArray(
        np.dstack([init_seq, init_qual]), 
        dims=('read', 'position', 'type'), 
        coords={'read': list(read_idx), 'position': list(position_idx), 'type': ['seq', 'quality']},
        attrs=new_attributes
    )
  
    for o in objects:
        overlapping_reads = list(set(o.read.values) & read_idx)
        overlapping_pos = list(set(o.position.values) & position_idx)
        new_xar.loc[overlapping_reads, overlapping_pos] = o.loc[overlapping_reads, overlapping_pos]
          
    return new_xar
        
       
    
    








# def merge(objects, compat='no_conflicts', join='outer'):
#     # HACKILY convert object datatype to xr2
#     hacky_types = [str(type(o)) for o in objects]

#     conversions = [
#         'Dataset' if 'core.dataset.Dataset' in ht else 'DataArray' if 'core.dataarray.DataArray' in ht else None
#         for ht in hacky_types
#     ]

#     new_objects = [getattr(xr2, c)(objects[i]) if c is not None else objects[i] for i, c in enumerate(conversions)]
#     merged_result = xr2.merge(new_objects, compat, join)
#     new_type = str(type(merged_result))

#     if 'core.dataset.Dataset' in new_type:
#         merged_result = xr.Dataset(merged_result)
#     elif 'core.dataarray.DataArray' in new_type:
#         merged_result = xr.DataArray(merged_result)

#     return merged_result


