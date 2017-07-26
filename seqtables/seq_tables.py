from __future__ import absolute_import
from ..internals import xarray_extensions
import xarray as xr
from ..internals import seq_df_to_datarray, _seqs_to_datarray


def df_to_dataset(df, seq_type, index, user_attrs={}, ref_name='', ref_to_pos_dict={}):
    ignore_ref_col = True if ref_name else False
    arrs, attrs = seq_df_to_datarray(df, seq_type, index, ignore_ref_col=ignore_ref_col, ref_name=ref_name, ref_to_pos_dict=ref_to_pos_dict)
    attrs['user'] = user_attrs
    return xr.Dataset(data_vars=arrs, attrs=attrs)


def seqs_to_dataset(
    seq_list, quality_score_list=None, ref_name='', pos=1, index=None, seq_type=None,
    phred_adjust=33, null_qual='!', encode_letters=True, user_attrs={}
):
    arrs, attrs = _seqs_to_datarray(
        seq_list, quality_score_list, ref_name, pos, index, seq_type,
        phred_adjust, null_qual, encode_letters
    )
    attrs['user'] = user_attrs

    return xr.Dataset(data_vars=arrs, attrs=attrs)


class SeqTable(xr.Dataset):
    def __init__(
        self, seq_list, quality_score_list=None, ref_name='', pos=1, index=None, seq_type=None,
        phred_adjust=33, null_qual='!', default_ref_name='', encode_letters=True
    ):
        arrs, attrs = _seqs_to_datarray(
            seq_list, quality_score_list, ref_name, pos, index, seq_type,
            phred_adjust, null_qual, encode_letters
        )

        return super(SeqTable, self).__init__(data_vars=arrs, attrs=attrs)