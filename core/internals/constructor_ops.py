from __future__ import absolute_import

import numpy as np
import pandas as pd
import xarray as xr
import warnings
from collections import defaultdict
from ..utils.alphabets import dna_alphabet, aa_alphabet, all_dna, all_aa
from .sam_to_arr import df_to_algn_arr


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
    Stupid convenience function to make sure sequences are returned as np array
    """
    if hasattr(values, 'values'):
        # its probably a dataframe/series
        # gotcha assumptions??? probably...
        return np.array(values.values, dtype='S')
    else:
        return np.array(values, dtype='S')


def _seq_df_to_datarray(
    df, seq_type, index,
    map_cols={'ref': 'rname', 'seqs': 'seq', 'quals': 'qual', 'cigar': 'cigar', 'pos': 'pos'},
    ref_pos_names={},
    ignore_ref_col=False,
    ref_name='',
    ref_to_pos_dict={}
):
    """

    """
    ref_pos_names = defaultdict(list, ref_pos_names)

    assert 'seqs' in map_cols, 'Error you must provide the column name for sequences'

    if 'cigar' not in map_cols:
        # no need to do any alignment, juse use seq to dtarr func
        return _seqs_to_datarray(
            df[map_cols['seqs']].values,
            df[map_cols['quals']].values if 'quals' in map_cols else None,
            pos=df[map_cols['pos']].values if 'pos' in map_cols else 1,
            index=list(index)
        )

    return _algn_seq_to_datarray(
        ref_name,
        seq_type,
        phred_adjust=33,
        data=[
            df[map_cols['seqs']].values,
            df[map_cols['quals']].values if 'quals' in map_cols else np.array([]),
            df[map_cols['pos']].values,
            df[map_cols['cigar']].values,
            np.array(list(index))
        ],
        ref_to_pos_dim=ref_to_pos_dict[ref_name] if ref_name in ref_to_pos_dict else None
    )


def _algn_seq_to_datarray(
    ref_name, seq_type, phred_adjust, data, ref_to_pos_dim=None
):
    """
        Create sets of xarray dataarrays from the variable data
        data is assumed to be 5xN matrix where each element in data are as follows:
            1. sequences
            2. qualities
            3. pos start
            4. cigar strings
    """
    # if data[-1] is None:
    #     index = np.arange(data[0].shape[0]) # np.array([])
    # elif isinstance(data[-1], list):
    #     index = np.array(data[-1])
    # else:
    #     index = data[-1]

    # add in gaps and remove indels in cython fnc

    aligned_arrs = df_to_algn_arr(*data, edge_gap=ord('$'))  # , edgeGap='$')

    has_quality = data[1].shape[0] > 0

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
        aligned_arrs[3],
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
        ins_data[:, 1] -= phred_adjust

    metadata = {
        'seq_type': seq_type,
        'fillna_val': fillna_val,
        'has_quality': has_quality,
        'insertions': pd.DataFrame(
            ins_data,
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
        'seqtable': metadata
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
    if isinstance(pos, int):
        pos_arr = np.arange(pos, pos + seq_arr.shape[1])
    elif isinstance(pos, list) or isinstance(pos, np.array):
        pos_arr = list(pos.copy())
        for i, p in enumerate(range(len(pos), seq_arr.shape[1])):
            # add extra values for pos
            warnings.warn('Warning adding additional positions for reference: ' + ref_name)
            pos_arr.append(pos_arr[-1] + i + 1)
        pos_arr = np.array(pos_arr)
    else:
        raise Exception('Error invalid type for position')

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