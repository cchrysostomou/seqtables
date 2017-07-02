from __future__ import absolute_import

import numpy as np
import xarray as xr
import warnings
from collections import defaultdict
from ..utils.seq_table_util import dna_alphabet, aa_alphabet


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
    sub_seq = ''.join(sub_seq)
    unique_letters = set(list(sub_seq.lower()))
    if len(unique_letters - set(list('actgn'))) == 0:
        # the first set of sequences had only actgn letters so we assume its a nucleotide sequence
        return 'NT'
    elif len(unique_letters - set(dna_alphabet)) == 0:
        # the first set of sequences have ACTGN letters and also letters that could refer to DEGENERATE bases. Or they could be a constricted set of AA residues so we arent sure, but default to nucleotide
        warnings.warn('The provided sequences appear to only have letters pertaining to DNA and degenerage DNA bases. So we are assuming that the provided sequences are DNA sequences. If this is incorrect, please manually define the seq_type when initializing the sequence table or change its sequence type via `change_seq_type` function')
        return 'NT'
    elif len(unique_letters - set(aa_alphabet)) == 0:
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


def _seq_df_to_dataarray(
    df,
    map_cols={'ref': 'rname', 'seqs': 'seqs', 'quals': 'quals', 'cigar': 'cigar'},
    ref_pos_names={}
):
    """

    """
    ref_pos_names = defaultdict(ref_pos_names, [])
    assert 'seqs' in map_cols, 'Error you must provide the column name for sequences'


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
        ).view(np.uint8)
        has_quality = True
        qual_arr -= phred_adjust
        assert qual_arr.shape == seq_arr.shape, \
            'Error the shape of the quality scores does not match the shape of the sequence scores: seq shape: \
            {0}, qual shape {1}'.format(
                seq_arr.shape,
                qual_arr.shape
            )
    else:
        has_quality = False
        qual_arr = np.array([]).reshape(*(0, 0))

    prefix = ref_name + '.' if ref_name else ''

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

    seq_xr = xr.DataArray(
        seq_arr,
        dims=[prefix + 'read', prefix + 'position'],
        coords={prefix + 'position': pos_arr, prefix + 'read': index} if index else {prefix + 'position': pos_arr}
    )

    seq_xr = xr.DataArray(
        seq_arr,
        dims=[prefix + 'read', prefix + 'position'],
        coords={prefix + 'position': pos_arr, prefix + 'read': index} if index else {prefix + 'position': pos_arr}
    )

    if has_quality:
        qual_xr = xr.DataArray(
            qual_arr,
            dims=[prefix + 'read', prefix + 'position'],
            coords={prefix + 'position': pos_arr, prefix + 'read': index} if index else {prefix + 'position': pos_arr}
        )
    else:
        qual_xr = xr.DataArray(qual_arr)

    metadata = {
        'seq_type': seq_type,
        'fillna_val': fillna_val,
        'has_quality': has_quality,
        'references': {
            ref_name: ['position', 'read']
        },
        'phred_adjust': phred_adjust
    }

    attrs = {
        'seqtable': metadata
    }

    return {
        prefix + 'sequence_table': seq_xr,
        prefix + 'quality_table': qual_xr,
        prefix + 'insertion_table': [],
    }, attrs
