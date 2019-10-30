from __future__ import absolute_import
import numbers
import numpy as np
import pandas as pd
import xarray as xr
import warnings
from collections import defaultdict
from seqtables.core.utils.alphabets import dna_alphabet, aa_alphabet, all_dna, all_aa
from seqtables.core.internals.sam_to_arr import df_to_algn_arr
import copy


def trim_str(seq, pos, minP, maxP, null_let):    
    s1 = max(0, minP - pos)
    frontLetters = null_let * max(0, pos - minP)
    
    seq = frontLetters + seq[s1:]
    pos = minP

    endLetters = null_let * max(0, (maxP - (pos + len(seq) - 1)))

    seq += endLetters

    s2 = min(maxP - minP + 1, len(seq))

    return seq[:s2]    
    

def strseries_to_bytearray(series, fillvalue='N', use_encoded_value=True):
    encode_type = 'S' if use_encoded_value else 'U'
    seq_arr = np.array(
        list(series), dtype=encode_type
    ).view(encode_type + '1').reshape((series.size, -1))
    if fillvalue != '':
        seq_arr[seq_arr.view(np.uint8) == 0] = fillvalue
    return seq_arr


def guess_seqtype(seq_data):
    """
    Predict whether sequences are nucleotide or protein

    Args:
        seqs (list of str): sets of sequences in table


    """
    # attempt to guess the sequence type
    check_sequences = min(len(seq_data), 1000)

    try:
        if hasattr(seq_data, 'iloc'):
            sub_seq = list(seq_data.values[:check_sequences].astype('U'))
        elif hasattr(seq_data, 'dtype'):
            sub_seq = list(seq_data[:check_sequences].astype('U'))
        else:
            sub_seq = list(seq_data[:check_sequences])
    except Exception as e:
        print(e)
        raise Exception("Error: cannot determine seq_type given the provided seqdata. Either manually define the sequence type or provide seq data as a list")
    # print(sub_seq)
    sub_seq = ''.join(sub_seq).upper()
    unique_letters = set(list(sub_seq))

    if len(unique_letters - set(list(dna_alphabet) + ['$', '-'])) == 0:
        # the first set of sequences had only actgn letters so we assume its a nucleotide sequence
        return 'NT'
    elif len(unique_letters - set(all_dna)) == 0:
        # the first set of sequences have ACTGN letters and also letters that could refer to DEGENERATE bases. Or they could be a constricted set of AA residues so we arent sure, but default to nucleotide
        warnings.warn('The provided sequences appear to only have letters pertaining to DNA and degenerage DNA bases. So we are assuming that the provided sequences are DNA sequences. If this is incorrect, please manually define the seq_type when initializing the sequence table or change its sequence type via `change_seq_type` function')
        return 'NT'
    elif len(unique_letters - set(all_aa + ['$', '-'])) == 0:
        # the first set of sequences have letters that only correspond to AA residues
        return 'AA'
    else:
        # the first set of sequences have letters that arent part of both DNA and AA values, so default to AA
        warnings.warn('The provided sequences appear to have letters outside the known/expected AA and NT nomenclature. The sequence type will be defaulted to AA sequences. If this is incorrect, please manually define the seq_type when initializing the sequence table or change its sequence type via `change_seq_type` function')
        return 'AA'


def list_to_arr(values):
    """
        Convenience function to make sure sequences are returned as np array
    """
    if hasattr(values, 'values'):
        # its probably a dataframe/series
        # gotcha assumptions??? probably...
        return np.array(values.values, dtype='S')
    else:
        return np.array(values, dtype='S')


def _seq_df_to_datarray(
    df, seq_type, index,
    map_cols = {},
    ref_pos_names={},
    ignore_ref_col=False,
    ref_name='',
    ref_to_pos_dict={},
    min_pos=-1,
    max_pos=-2
):
    """

    Converts a dataframe into seqtables format. It is assumed that the dataframe has columns which follows sam file specifications and defines the:
        1. The sequence
        2. The reference name
        3. Quality score
        4. Cigar string
        5. Start position
    
    Args:
        df (dataframe): dataframe converted from a sam file
        map_cols (dict): dict defining how to map column names to the expected fields
        ref_pos_names (dict): if defined, defines which references we want to define
        ignore_ref_col (bool): If true, then will not use the reference column to define sequences
        ref_name (string): Define the sequences with a reference
        min_pos (default = -1): If > -1 then defines the minimum position allowed for alignment, all other aligned positions before min_pos will be trimmed

            .. note::

                For example, let an alignment start at position 1 as follows:
                    AACGA 1
                
                If min_pos is 3 then the following sequence is returned
                    CGA 
        
        max_pos (default = -2): If > 0 then defines the maximum position allowed for alignment
        ref_to_pos_dict (dict): Allows user to custom name the 'position' dimension. If empty, then dimension is just 'position'
    
    Returns: 
        xarray
        attributes for xarray

    """
    expected_map_cols={'ref': 'rname', 'seqs': 'seq', 'quals': 'qual', 'cigar': 'cigar', 'pos': 'pos'}
    expected_map_cols.update(map_cols)
    map_cols = copy.deepcopy(expected_map_cols)
    ref_pos_names = defaultdict(list, ref_pos_names)
    
    # store these as an attribute
    metadata_columns = [
        c for c in df.columns
        if c not in list(map_cols.values())
    ]
    

    assert 'seqs' in map_cols, 'Error you must provide the column name for sequences'

    # print(map_cols)
    has_quality = 'quals' in map_cols and map_cols['quals'] in df.columns
    # print(df.columns, map_cols['quals'], has_quality)
    has_cigar = 'cigar' in map_cols and map_cols['cigar'] in df.columns

    assert 'pos' in map_cols
    
    # print(df[map_cols['ref']].value_counts())
    # print('yadda')
    # print(df.head())

    if map_cols['pos'] not in df.columns:
        warnings.warn('Position column not found, automatically assuming sequences are aligned at position 1')
        df[map_cols['pos']] = 1

    assert 'seqs' in map_cols and map_cols['seqs'] in df.columns, 'Error cannot find the column corresponding to sequences: ' + map_cols['seqs'] + '. Columms in df: ' + ':'.join(df.columns)
    
    if has_cigar is False:                
        if min_pos < 0:
            min_pos = df[map_cols['pos']].min()        
        if max_pos < 0:
            max_pos =  df[[map_cols['seqs'], map_cols['pos']]].apply(lambda x: x[1] - 1 + len(x[0]), axis=1).max()
        # print(min_pos, max_pos)
        # no need to do any alignment, juse use seq to dtarr func
        return _seqs_to_datarray(
            df[[map_cols['seqs'], map_cols['pos']]].apply(lambda x: trim_str(x[0], x[1], min_pos, max_pos, '$'), axis=1).values,
            df[[map_cols['quals'], map_cols['pos']]].apply(lambda x: trim_str(x[0], x[1], min_pos, max_pos, '!'), axis=1).values if has_quality is True else None,            
            pos=min_pos,            
            index=pd.Index(index)
        )

    if has_quality is False:
        df[map_cols['quals']] = ''
    
    if len(metadata_columns) > 0:
        metadata_df = pd.DataFrame(df[metadata_columns], index=index)
    else:
        metadata_df = pd.DataFrame()

    return _algn_seq_to_datarray(        
        ref_name,
        seq_type,
        phred_adjust=33,
        data=[
            df[map_cols['seqs']].values,
            df[map_cols['quals']].values,  # if has_quality is True else None, # np.array([]),
            df[map_cols['pos']].astype(np.int64).values,
            df[map_cols['cigar']].values,
            np.array(list(index))
        ],
        has_quality=has_quality,
        min_pos=min_pos,
        max_pos=max_pos,
        ref_to_pos_dim=ref_to_pos_dict[ref_name] if ref_name in ref_to_pos_dict else None,
        read_info=metadata_df
    )


def _algn_seq_to_datarray(
    ref_name, seq_type, phred_adjust, data, has_quality, ref_to_pos_dim=None, min_pos=-1, max_pos=-2, edge_gap='$', null_quality='!', read_info=pd.DataFrame()
):
    """
        Create sets of xarray dataarrays from the variable data
        data is assumed to be 5xN matrix where each element in data are as follows:
            1. sequences
            2. qualities (this column is ignored in the output from df_to_algn_arr if has_quality is False)
            3. pos start
            4. cigar strings
    """
    # if data[-1] is None:
    #     index = np.arange(data[0].shape[0]) # np.array([])
    # elif isinstance(data[-1], list):
    #     index = np.array(data[-1])
    # else:
    #     index = data[-1]
    # print('yadda', data[0].shape)
    if data[0].shape[0] == 0:
        print('data is removing')
        return xr.DataArray([]), {}            
    # print('HEY ITS A SHAPE', data[0].shape)
    # import pickle
    # pickle.dump(data, open('saveapickle.pkl', 'wb'))


    # add in gaps and remove indels in cython fnc
    aligned_arrs = df_to_algn_arr(*data, edge_gap=ord(edge_gap), null_quality=ord(null_quality), min_pos=min_pos, max_pos=max_pos)  # , edgeGap='$')

    # has_quality = data[1].shape[0] > 0

    seq_arr = aligned_arrs[0].astype('S1')
    # index = data[-1]
    if len(data) == 0:
        has_quality = False
        qual_arr = np.array([]).reshape(*(0, 0))
    else:
        # qual_arr = aligned_arrs[1].view(np.int8) - 33
        qual_arr = aligned_arrs[1].view('S1')  # (np.int8) - 33

    # ### TO DO: FIX PREFIX CALL, SHOULD WE INCLUDE A PREFIX OR NOT? CURRENTLY CHOOSING NOT TO ###
    prefix = ref_name + '_' if ref_name else ''
    prefix = ''

    pos_arr = aligned_arrs[2]
    # print(aligned_arrs[3])

    position_dim = prefix + 'position' if ref_to_pos_dim is None else ref_to_pos_dim

    if has_quality:
        # create a dataarray for aligned sequences
        seq_xr = xr.DataArray(
            np.dstack([seq_arr, qual_arr]),
            dims=[prefix + 'read', position_dim, 'type'],
            coords={
                position_dim: pos_arr,
                prefix + 'read': aligned_arrs[5],  # index if index.shape[0] > 0 else np.range(1, 1 + seq_arr.shape[0])
                'type': ['seq', 'quality']
            }
        )
    else:
        # create a dataarray for aligned sequences
        seq_xr = xr.DataArray(
            seq_arr.reshape(seq_arr.shape[0], seq_arr.shape[1], 1),
            dims=[prefix + 'read', position_dim, 'type'],
            coords={
                position_dim: pos_arr,
                prefix + 'read': aligned_arrs[5],  # index if index.shape[0] > 0 else np.range(1, 1 + seq_arr.shape[0])
                'type': ['seq']
            }
        )

    # create a dataarray for quality scores
    # if has_quality:
    #     qual_xr = xr.DataArray(
    #         qual_arr,
    #         dims=[prefix + 'read', position_dim],
    #         coords={
    #            position_dim: pos_arr,
    #            prefix + 'read': aligned_arrs[5] # index if index.shape[0] > 0 else np.range(1, 1 + seq_arr.shape[0])
    #         }
    #     )
    # else:
    #     qual_xr = xr.DataArray(np.array([]).reshape(*(0, 0)))

    fillna_val = 'N' if seq_type == 'NT' else 'X'

    # create a dataarray for insertions
    inserted_base_index = pd.MultiIndex.from_tuples(
        aligned_arrs[3] or [(None, None, None)],
        names=(
            'read_ins',
            'position_ins',
            'loc_ins'
        )
        # names=(
        #    'insertions_{0}{2}{1}'.format(prefix, 'read', '_' if not prefix else ''),
        #    'insertions_{0}{1}refpos'.format(prefix, '_' if not prefix else ''),  # base position with respect to reference
        #    'insertions_{0}{1}inspos'.format(prefix, '_' if not prefix else '')  # specific position of insertion
        # )
    )

    # insertions = xr.DataArray(
    #     aligned_arrs[4],
    #     coords={
    #     #    'insertions': inserted_base_index,
    #     #    'seq_let': (['letter', 'quality'])
    #           prefix + 'insertions': inserted_base_index,
    #          'seq_let': (['letter', 'quality'])
    #     },
    #     dims=[prefix + 'insertions', 'seq_let']
    #     # dims = ['insertions', 'seq_let']
    # )

    ins_data = aligned_arrs[4].view(np.uint8)
    
    if has_quality:
        if ins_data.shape[0] > 0:
            ins_data[:, 1] -= phred_adjust
        else:
            ins_data = ins_data.reshape(0, 0)

    metadata = {
        'seq_type': seq_type,
        'fillna_val': fillna_val,
        'has_quality': has_quality,
        'insertions': pd.DataFrame(
            ins_data if ins_data.shape[0] > 0 else None,
            index=inserted_base_index,
            columns=['seq', 'quality'] if has_quality else ['seq'],
        ),
        'references': [ref_name],

        # 'references': {
        #     ref_name: {
        #         'dimension_names': {
        #             'position': position_dim,
        #             'read': prefix + 'read',
        #             'insertion_positions': prefix + 'insertion_table_pos_data',
        #             'insertion_positions_multiindex': inserted_base_index.names
        #         },
        #         'table_names': {
        #             'seq': prefix + 'sequence_table',
        #             'qual': prefix + 'quality_table',
        #             'ins':  prefix + 'insertion_table'
        #         }
        #     }
        # },
        'phred_adjust': phred_adjust
    }

    attrs = {
        'seqtable': metadata,
        'read_info': read_info
    }

    return seq_xr, attrs

    # return {
    #     # prefix + 'sequence_table': seq_xr,
    #     # prefix + 'quality_table': qual_xr,
    #     # prefix + 'insertion_table': insertions,
    #     'sequence_table': seq_xr,
    #     'quality_table': qual_xr,
    #     'insertion_table': insertions
    # }, attrs


def _seqs_to_datarray(
    seq_list, quality_score_list=None, ref_name='', pos=1, index=None, seq_type=None,
    phred_adjust=33, null_qual='!', encode_letters=True
):
    """
        For a given reference convert a table of sequences (aligned to the reference), quality scores (optional),
        and insertions (optional) into an xarray dataset
    """

    # null_qual = null_qual
    if seq_type is None:
        seq_type = guess_seqtype(seq_list)
    elif seq_type not in ['AA', 'NT']:
        raise Exception('You defined seq_type as, {0}. We only allow seq_type to be "AA" or "NT"'.format(seq_type))
    else:
        # user appropriately defined the type of sequences in the list
        seq_type = seq_type

    seq_list = list_to_arr(seq_list)

    fillna_val = 'N' if seq_type == 'NT' else 'X'

    # create numpy array of sequences
    seq_arr = strseries_to_bytearray(
        seq_list, fillna_val, encode_letters
    )

    if quality_score_list is not None:
        # create numpy array of quality scores
        # create a quality array
        qual_list = list_to_arr(quality_score_list)
        qual_arr = strseries_to_bytearray(
            qual_list, null_qual, encode_letters
        ).view('S1')  # view(np.uint8)
        has_quality = True
        # qual_arr -= phred_adjust
        assert qual_arr.shape == seq_arr.shape, \
            'Error the shape of the quality scores does not match the shape of the sequence scores: seq shape: \
            {0}, qual shape {1}'.format(
                seq_arr.shape,
                qual_arr.shape
            )
    else:
        has_quality = False
        qual_arr = np.array([]).reshape(*(0, 0))

    # ### TO DO: FIX PREFIX CALL, SHOULD WE INCLUDE A PREFIX OR NOT? CURRENTLY CHOOSING NOT TO ###
    prefix = ref_name + '_' if ref_name else ''
    prefix = ''

    position_dim = prefix + 'position'  # if ref_to_pos_dim is None else ref_to_pos_dim

    # create dimensions    
    if isinstance(pos, numbers.Number):
        pos = int(pos)
        pos_arr = np.arange(pos, pos + seq_arr.shape[1])
    else:
        pos_arr = list(copy.deepcopy(pos))
        for i, p in enumerate(range(len(pos), seq_arr.shape[1])):
            # add extra values for pos
            warnings.warn('Warning adding additional positions for reference: ' + ref_name)
            pos_arr.append(pos_arr[-1] + i + 1)
        pos_arr = np.array(pos_arr)
    
    # print(pos)
    # pos = int(pos)
    
    # if isinstance(pos, int):
        
    # elif isinstance(pos, list) or isinstance(pos, np.ndarray):
    #     pos_arr = list(copy.deepcopy(pos))
    #     for i, p in enumerate(range(len(pos), seq_arr.shape[1])):
    #         # add extra values for pos
    #         warnings.warn('Warning adding additional positions for reference: ' + ref_name)
    #         pos_arr.append(pos_arr[-1] + i + 1)
    #     pos_arr = np.array(pos_arr)
    # else:
    #     raise Exception('Error invalid type for position')

    if has_quality:
        # create a dataarray for aligned sequences
        seq_xr = xr.DataArray(
            np.dstack([seq_arr, qual_arr]),
            dims=[prefix + 'read', position_dim, 'type'],
            coords={
                position_dim: pos_arr,
                prefix + 'read': index if not(index is None) and index.shape[0] > 0 else np.arange(1, 1 + seq_arr.shape[0]),
                'type': ['seq', 'quality']
            }
        )
    else:
        # create a dataarray for aligned sequences
        seq_xr = xr.DataArray(
            seq_arr.reshape(seq_arr.shape[0], seq_arr.shape[1], 1),
            dims=[prefix + 'read', position_dim, 'type'],
            coords={
                position_dim: pos_arr,
                prefix + 'read': index if not(index is None) and index.shape[0] > 0 else np.arange(1, 1 + seq_arr.shape[0]),
                'type': ['seq']
            }
        )

    # seq_xr = xr.DataArray(
    #     seq_arr,
    #     dims=[prefix + 'read', prefix + 'position'],
    #     coords={prefix + 'position': pos_arr, prefix + 'read': index} if index else {prefix + 'position': pos_arr}
    # )

    # seq_xr = xr.DataArray(
    #     seq_arr,
    #     dims=[prefix + 'read', prefix + 'position'],
    #     coords={prefix + 'position': pos_arr, prefix + 'read': index} if index else {prefix + 'position': pos_arr}
    # )

    # if has_quality:
    #     qual_xr = xr.DataArray(
    #         qual_arr,
    #         dims=[prefix + 'read', prefix + 'position'],
    #         coords={prefix + 'position': pos_arr, prefix + 'read': index} if index else {prefix + 'position': pos_arr}
    #     )
    # else:
    #     qual_xr = xr.DataArray(qual_arr)

    metadata = {
        'seq_type': seq_type,
        'fillna_val': fillna_val,
        'has_quality': has_quality,
        'insertions': pd.DataFrame(),
        # 'references': {
        #    ref_name: ['position', 'read']
        # },
        'phred_adjust': phred_adjust,
        'references': [ref_name]
    }

    attrs = {
        'seqtable': metadata
    }

    return seq_xr, attrs

    # return {
    #     prefix + 'sequence_table': seq_xr,
    #     prefix + 'quality_table': qual_xr,
    #     prefix + 'insertion_table': [],
    # }, attrs
