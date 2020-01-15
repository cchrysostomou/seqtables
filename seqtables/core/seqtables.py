from __future__ import absolute_import
# py2/py3 compatibility
from builtins import (bytes, str, open, super, range,
                      zip, round, input, int, pow, object)
# from ..internals import xarray_extensions
import xarray as xr
from seqtables.core.internals import _seq_df_to_datarray, _seqs_to_datarray
from seqtables.core.seq_logo import draw_seqlogo_barplots, get_bits, get_plogo, shannon_info, relative_entropy
from seqtables.xarray_mods import st_merge
from seqtables.core import numpy_ops
from seqtables.core.utils.custom_sam_utils import read_sam
import warnings
import copy
import numpy as np
import pandas as pd
# import scipy
from six import string_types
from collections import defaultdict
import itertools
from orderedset import OrderedSet

def df_to_dataarray(
    df, 
    seq_type, 
    index, 
    map_cols={'ref': 'rname', 'seqs': 'seq', 'quals': 'qual', 'cigar': 'cigar', 'pos': 'pos'},
    ref_pos_names={},
    ignore_ref_col=False,
    ref_name='',
    ref_to_pos_dict={},
    min_pos=-1,
    max_pos=-2,
    user_attrs={}
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
    # convert '\*' fields into np.nan
    if map_cols['ref'] in df.columns:
        df[map_cols['ref']].replace(r'\*', np.nan, inplace=True)
    if map_cols['cigar'] in df.columns:
        df[map_cols['cigar']].replace(r'\*', np.nan, inplace=True)
    has_null_field = df.isnull().sum(axis=1)
    remove_rows = (has_null_field > 0).sum()
    if remove_rows > 0:
        warnings.warn('Warning, dropping {0} rows because they contained null fields either in their reference or cigar'.format(remove_rows))
        df = df[has_null_field == 0]
    
    # print(df.columns)
    ignore_ref_col = True if ref_name else False
    arrs, attrs = _seq_df_to_datarray(
        df, seq_type, index, ignore_ref_col=ignore_ref_col, ref_name=ref_name, ref_to_pos_dict=ref_to_pos_dict,
        min_pos=min_pos, max_pos=max_pos, map_cols=map_cols
    )

    # return xr.Dataset(data_vars=arrs, attrs=attrs)
    return xr.DataArray(arrs, attrs=attrs, name=ref_name if ref_name else None)


def seqs_to_datarray(
    seq_list, quality_score_list=None, ref_name='', pos=1, index=None, seq_type=None,
    phred_adjust=33, null_qual='!', encode_letters=True, user_attrs={}
):
    """
        Converts a list of sequences and qualities into an XARRAY object (not a seqtable)

        .. important:: cigar string

            Assumes there are no insertions for this list of sequences and they are already aligned to one another

    """
    arrs, attrs = _seqs_to_datarray(
        seq_list, quality_score_list, ref_name, pos, index, seq_type,
        phred_adjust, null_qual, encode_letters
    )
    attrs['user_defined'] = user_attrs

    # return xr.Dataset(data_vars=arrs, attrs=attrs)
    return xr.DataArray(arrs, attrs=attrs, name=ref_name if ref_name else None)


def merge_seqs(*args, **kwargs):
    xarr = st_merge.st_merge_arrays(*args, **kwargs)
    new_st_tab = SeqTable(xarr)
    # new_st_tab.update_attributes()
    return new_st_tab


class SeqTable(xr.DataArray):
    """
    Class for viewing aligned sets of DNA or AA sequences. This will take in a list of sequences that are presumed to be aligned to one another, and
    convert the sequences into an xarray dataset (sets of numpy arrays). In additiona to the sequence, quality scores for each base can be provided in addition to sequences.

    The xarray datarray will be structured as follows:
        1. Dimension 1: sequences/seqids/header sequences
        2. Dimension 2: position array
        3. Dimension 3: [seq or quality]

    Args:
        seqdata (Series, or list of strings): List containing a set of sequences aligned to one another
        qualitydata (Series or list of quality scores, default=None): If defined, then user is passing in quality data along with the sequences)
        reference_positions (int, list, array, series, dict): Explicitly define where the aligned sequences start with respect to some reference frame (i.e. start > 2 means sequences start at position 2 not 1)

            .. note:: dict

                If reference_positions is dict, then it assumes a unique set of coordinates for each reference



        index (list of values defining the index, default=None):

            .. note::Index=None

                If None, then the index will result in default integer indexing by pandas.

        seq_type (string of 'AA' or 'NT', default='NT'): Defines the format of the data being passed into the dataframe
        phred_adjust (integer, default=33): If quality data is passed, then this will be used to adjust the quality score (i.e. Sanger vs older NGS quality scorning)
        encode_letters (bool, default=True):

            If True, then strings will be encoded based on setting define din encoding (utf-8 encoding results is 1 byte representation per character rather than 4 bytes)
            If False, then strings should be represented as str

        encoding (str, default='utf-8'): If encode_letters is true, then this will encode strings using this setting

    Attributes:
        seq_df (Dataframe): Each row in the dataframe is a sequence. It will always contain a 'seqs' column representing the sequences past in. Optionally it will also contain a 'quals' column representing quality scores
        seq_table (Dataframe): Dataframe representing sequences as characters in a table. Each row in the dataframe is a sequence. Each column represents the position of a base/residue within the sequence. The 4th position of sequence 2 is found as seq_table.ix[1, 4]
        qual_table (Dataframe, optional): Dataframe representing the quality score for each character in seq_table

    Examples:
        >>> sq = seqtables.SeqTable(['AAA', 'ACT', 'ACA'])
        >>> sq.hamming_distance('AAA')
        >>> sq = read_fastq('fastqfile.fq')
    """

    @staticmethod
    def from_df(*args, **kwargs):
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
        new_st = SeqTable(df_to_dataarray(*args, **kwargs))
        # new_st.update_attributes()
        return new_st

    @staticmethod
    def from_list(*args, **kwargs):
        new_st = SeqTable(seqs_to_datarray(*args, **kwargs))
        # new_st.update_attributes()
        return new_st

    @staticmethod
    def from_sam(sam_file_location, std_fields_keep=['header', 'flag', 'rname', 'pos', 'cigar', 'seq', 'qual'], opt_fields_keep=['XN', 'XM', 'MD'], nrows=None, chunks=None, indexing_dict = None, ignore_quotes=True, comment_out_letter=False, min_pos=-1, max_pos=-2, seq_type='NT'):
        """
            Returns an iterator that reads through a sam file using pysam and then returns results in seqtable format
        """
        samfile_reader = read_sam(sam_file_location, std_fields_keep, opt_fields_keep, nrows,chunks,indexing_dict,ignore_quotes,comment_out_letter)
        
        for sam_df in samfile_reader:
            sam_df = sam_df[sam_df['rname'] != '*']            
            st = SeqTable.from_df(sam_df[['seq', 'rname', 'qual', 'pos', 'cigar']], index=sam_df['header'],seq_type=seq_type, min_pos=min_pos, max_pos=max_pos)
            if st.shape[0] > 0:
                yield st

    @staticmethod
    def from_pysam(alignment_file, chunks=None, fetch_args=[], fetch_kwargs={}, seq_type='NT', store_additional_features=['mapping_quality'], min_mapping_quality=None, min_pos=-1, max_pos=-2):
        """
            Returns an iterator that reads through a sam file using pysam and then returns results in seqtable format

            Args:
                alignment_file: iterator to pysam alignment file object
                chunks (int): number of reads to import at a given time
                fetch_args (list): positional arguments passed into psyam.fetch
                fetch_kwargs (dict): keyword arguments passed into pysam.fetch

            Returns:
                Iterator to seqtable object                
        """
        try:
            samfile_reader = alignment_file.fetch(*fetch_args, **fetch_kwargs)
        except ValueError:
            samfile_reader = alignment_file

        if chunks is None:
            chunks = float('Inf')
                
        counter = 0
        data = []
        for read in samfile_reader:        
            if min_mapping_quality is not None and read.mapping_quality < min_mapping_quality:
                continue
            # read_info = [read.query_name, read.reference_name, read.seq, read.qual, read.pos, read.cigarstring]
            read_info = [read.query_name, read.reference_name, read.query_sequence, read.query_qualities, read.reference_start, read.cigarstring]
            for s in store_additional_features:
                read_info.append(getattr(read, s))
            data.append(read_info)
            counter += 1
            if counter == chunks:                    
                df = pd.DataFrame(data, columns=['header', 'rname', 'seq', 'qual', 'pos', 'cigar'] + store_additional_features).set_index('header')                                                
                # add 1 to the position because pysam treats positions from 0 index (converts position from bowtie to 0 base index)
                df['pos'] += 1
                # yield SeqTable.from_df(df, index=df.index, seq_type=seq_type, min_pos=min_pos, max_pos=max_pos) 
                st = SeqTable.from_df(df, index=df.index, seq_type=seq_type, min_pos=min_pos, max_pos=max_pos) 
                if st.shape[0] > 0:
                    yield st
                counter = 0
                data = []
            
        if len(data) > 0:
            df = pd.DataFrame(data, columns=['header', 'rname', 'seq', 'qual', 'pos', 'cigar'] + store_additional_features).set_index('header')
            # add 1 to the position because pysam treats positions from 0 index
            df['pos'] += 1
            # yield SeqTable.from_df(df, index=df.index, seq_type=seq_type, min_pos=min_pos, max_pos=max_pos) 
            st = SeqTable.from_df(df, index=df.index, seq_type=seq_type, min_pos=min_pos, max_pos=max_pos) 
            if st.shape[0] > 0:
                yield st
            counter = 0
            data = []

    def __init__(self, seq_list, *args, **kwargs):
        if isinstance(seq_list, list) or isinstance(seq_list, np.ndarray) or isinstance(seq_list, pd.Series):
            xarr = seqs_to_datarray(seq_list, *args, **kwargs)                        
            # super(SeqTable, self).__init__(xarr)  # data_vars=arrs, attrs=attrs)
            super().__init__(xarr)  # data_vars=arrs, attrs=attrs)
            # self.update_attributes()
        else:
            # super(SeqTable, self).__init__(seq_list, *args, **kwargs)
            super().__init__(seq_list, *args, **kwargs)

    # def update_attributes(self):
    #     self.phred_adjust = self.attrs['seqtable'].get('phred_adjust', 33)
    #     self.fill_na_val = self.attrs['seqtable'].get('fill_na', 'N')
    #     self.insertions = self.attrs['seqtable'].get('insertions')
    #     self.seq_type = self.attrs['seqtable'].get('seq_type')
    #     self.has_quality = 'quality' in self.type.values if hasattr(self, 'type') else False

    # def _copy_attributes(self, new):
    #     if hasattr(self, 'phred_adjust'):
    #         new.phred_adjust = self.phred_adjust
    #     if hasattr(self, 'fill_na_val'):
    #         new.fill_na_val = self.fill_na_val
    #     if hasattr(self, 'insertions'):
    #         new.insertions = self.insertions
    #     if hasattr(self, 'seq_type'):
    #         new.seq_type = self.seq_type
    #     if hasattr(self, 'has_quality'):
    #         new.has_quality = self.has_quality
    #     if hasattr(self, 'attrs'):
    #         new.attrs = copy.deepcopy(self.attrs)

    def _has_attrs(self):
        if 'seqtable' not in self.attrs:
            warnings.warn('This object does not appear to be a seqtable object. maybe its just a datarray object?')

    @property
    def loc(self):
        sliced = xr.DataArray.loc.fget(self)
        # self._copy_attributes(sliced)
        # TO DO: FILTER OUT INSERTIONS BASED ON THE READS, POSITIONS, AND DATATYPE RETURNED BY LOC
        return sliced

    def __getitem__(self, *args, **kwargs):
        sliced = xr.DataArray.__getitem__(self, *args, **kwargs)
        # TO DO: FILTER OUT INSERTIONS BASED ON THE READS, POSITIONS, AND DATATYPE RETURNED BY LOC
        return sliced

    def isel(self, *args, **kwargs):
        sliced = xr.DataArray.isel(self, *args, **kwargs)
        # TO DO: FILTER OUT INSERTIONS BASED ON THE READS, POSITIONS, AND DATATYPE RETURNED BY LOC
        return sliced

    def sel(self, *args, **kwargs):
        sliced = xr.DataArray.sel(self, *args, **kwargs)
        # TO DO: FILTER OUT INSERTIONS BASED ON THE READS, POSITIONS, AND DATATYPE RETURNED BY LOC
        return sliced

    @property
    def phred_adjust(self):
        self._has_attrs()
        return self.attrs.get('seqtable', {}).get('phred_adjust', 33)

    @property
    def fill_na_val(self):
        self._has_attrs()
        return self.attrs.get('seqtable', {}).get('fill_na_val', 'N')

    @property
    def null_qual(self):
        self._has_attrs()
        return self.attrs.get('seqtable', {}).get('null_qual', '!')

    @property
    def insertions(self):
        self._has_attrs()
        return self.attrs.get('seqtable', {}).get('insertions', None)

    @property
    def seq_type(self):
        self._has_attrs()
        return self.attrs.get('seqtable', {}).get('seq_type', None)

    @property
    def has_quality(self):
        self._has_attrs()
        if hasattr(self, 'type') is False:
            warnings.warn('This object does not appear to have a type attribute. We expected a dimension to be labelled as type. Maybe its just a datarray object?')
            return False
        return 'quality' in self.type.values

    @property
    def encoding_setting(self):
        self._has_attrs()
        return self.attrs.get('seqtable', {}).get('encoding_setting', (True, 'utf-8'))

    def get_sequences(self):
        """
            Return the letters for all sequences

            Returns:
                sequences (np.array)
        """
        return xr.DataArray(self.sel(type='seq'))

    def get_quality(self, as_num=True):
        """
            Return the quality scores for all sequences

            Args:
                as_num (bool, true): If true then it returns the quality scores as numbers

            Returns:
                quality (np.array)
        """
        if as_num:
            return xr.DataArray(
                self.sel(type='quality').values.view(np.uint8) - self.phred_adjust,
                dims=('read', 'position'),
                coords={'read': self.read, 'position': self.position}
            )
        else:
            return xr.DataArray(self.sel(type='quality'))

    def view_with_ins(self, positions=None, min_ins_count=0, ins_gap='-', return_as_dataframe=True, include_quality=False, lowercase_insertions=True):
        """
            Create a new stacked table where insertions are also represented as columns

            ..note:: performance

                The memory and required time for this operation has not been tested thoroughly yet
        """

        # insertion_table_name = xarr.attrs['seqtable']['references'][table_ref]['table_names']['ins']
        # sequence_table_name = xarr.attrs['seqtable']['references'][table_ref]['table_names']['seq']
        # coord_name = xarr.attrs['seqtable']['references'][table_ref]['dimension_names']['insertion_positions']
        # p1, p2 = xarr.attrs['seqtable']['references'][table_ref]['dimension_names']['insertion_positions_multiindex'][1:]
        # read_name = xarr.attrs['seqtable']['references'][table_ref]['dimension_names']['read']
        # pos_name = xarr.attrs['seqtable']['references'][table_ref]['dimension_names']['position']
        include_quality = include_quality is True and self.has_quality
        total_elements = self.insertions.shape[0]  # xarr[p1].shape[0]
        p1 = self.insertions.index.get_level_values(1)
        p2 = self.insertions.index.get_level_values(2)

        if min_ins_count > 0:
            # first collapse the two levels into their unique values, but also return the inverse of unique values so we can map the original index to its respective unique value
            un_vals, unique_idx, un_counts = np.unique(
                np.dstack([
                    p1,
                    p2
                ]).squeeze(),
                axis=0,
                return_counts=True,
                return_inverse=True
            )
            keep_these_rows = np.where(un_counts[unique_idx] > min_ins_count, True, False)
        else:
            keep_these_rows = np.array([True] * total_elements)

        # next identify all rows in the index that have insertions at specific positions which repeat more than min_ins_count
        if positions is None:
            positions = list(self.position.values)
        else:
            positions = list(positions)
        # print(positions)
        is_in_position_of_interest = np.in1d(p1, positions)
        # print(len(is_in_position_of_interest), p1)
        # print(self.insertions.iloc[is_in_position_of_interest, :])

        # finally we only want rows that are present in both filteres        
        final_rows_to_select = np.where(keep_these_rows & is_in_position_of_interest)[0]

        # filtered_insertion_table = xarr[insertion_table_name].loc[(slice(None), filtered_ins_index[:, 0], filtered_ins_index[:, 1]), :]
        filtered_insertion_table = self.insertions.iloc[final_rows_to_select, :]
        # print(filtered_insertion_table)
        if filtered_insertion_table.shape[0] == 0:
            seqs_with_ins = pd.DataFrame(
                self.loc[:, positions, 'seq'].values.view(np.uint8),
                columns=positions, index=self.read.values
            )
        else:
            ins_table_seq = filtered_insertion_table['seq'].unstack(level=(1, 2), fill_value=ord(ins_gap) if lowercase_insertions is False else ord(ins_gap) - 32)

            # seqs_with_ins = pd.concat([
            #     pd.DataFrame(
            #         self.loc[:, positions, 'seq'].values.view(np.uint8),
            #         columns=positions, index=self.read.values
            #     ),
            #     ins_table_seq
            # ], axis=1).fillna(ord(ins_gap)).astype(np.uint8)
            
            seqs_with_ins = pd.DataFrame(
                self.loc[:, positions, 'seq'].values.view(np.uint8),
                columns=pd.MultiIndex.from_product([positions, [0]]), index=self.read.values
            ).merge(
                ins_table_seq if lowercase_insertions is False else ins_table_seq + 32, 
                left_index=True, 
                right_index=True,
                how='left'
            ).dropna(how='all', axis=1).fillna(ord(ins_gap)).astype(np.uint8)

        # realign column sin table so that insertion bases are in the proper position with respect to referene positions
        cols = list(seqs_with_ins.columns)
        sorted_column_indicies = sorted(
            range(len(cols)),
            key=lambda k: self._sort_merged_columns(cols[k])
        )

        seqs_with_ins = seqs_with_ins.iloc[:, sorted_column_indicies]
        renamed_cols = self._make_positions_multiindex(seqs_with_ins.columns, names=['read_pos', 'loc_ins'])
        seqs_with_ins.columns = renamed_cols

        # now merge bases table with insertion table
        if include_quality is True:
            if filtered_insertion_table.shape[0] == 0:
                quals_with_ins = pd.DataFrame(
                    self.loc[:, positions, 'quality'].values.view(np.uint8) - self.phred_adjust,
                    columns=positions, index=self.read.values
                )
            else:
                ins_table_qual = filtered_insertion_table['quality'].unstack(level=(1, 2), fill_value=ord(self.null_qual) - self.phred_adjust)
                # quals_with_ins = pd.concat([
                #     pd.DataFrame(
                #         self.loc[:, positions, 'quality'].values.view(np.uint8) - self.phred_adjust,
                #         columns=positions, index=self.read.values
                #     ),
                #     ins_table_qual
                # ], axis=1).fillna(ord(self.null_qual) - self.phred_adjust).astype(np.uint8)
                quals_with_ins = pd.DataFrame(
                    self.loc[:, positions, 'quality'].values.view(np.uint8) - self.phred_adjust,
                    columns=pd.MultiIndex.from_product([positions, [0]]), index=self.read.values
                ).merge(
                    ins_table_qual,
                    left_index=True,
                    right_index=True,
                    how='left'
                ).dropna(how='all', axis=1).fillna(ord(self.null_qual) - self.phred_adjust).astype(np.uint8)

            quals_with_ins = quals_with_ins.iloc[:, sorted_column_indicies]
            quals_with_ins.columns = renamed_cols

            seqs_with_ins = pd.concat([seqs_with_ins, quals_with_ins], axis=1, keys=['seq', 'quality'])
        
        # if seqs_with_ins.empty:
        #     # its an empty dataframe, so need to add columns for accessing keys later on
        #     print('some positions', positions)
        #     print(self.loc[:, positions])
        #     print('seq_with_ins_before', seqs_with_ins)            
        #     seqs_with_ins = pd.DataFrame([] dtype=np.uint8, columns=['seq', 'quality'])
        #     print('seq_with_ins_after', seqs_with_ins)
        # try:
        if return_as_dataframe is True:
            return seqs_with_ins
        else:
            if include_quality is True:
                return xr.DataArray(
                    np.dstack([
                        seqs_with_ins['seq'].values,
                        seqs_with_ins['quality'].values + self.phred_adjust,
                    ]).view('S1'),
                    dims=('read', 'position', 'type'),
                    coords={
                        'read': seqs_with_ins.index,
                        'position': renamed_cols,
                        'type': ['seq', 'quality']
                    }
                )
            else:
                return xr.DataArray(
                    seqs_with_ins.values.view('S1'),
                    dims=('read', 'position'),
                    coords={
                        'read': seqs_with_ins.index,
                        'position': renamed_cols,
                    }
                )
        # except Exception as e:
        #     print('DEBUGGING', return_as_dataframe, include_quality)
        #     print(seqs_with_ins)
        #     print(str(e))
        #     raise Exception('THE STEP IN VIEW WITH INS FAILED!!')

    def slice_sequences(self, positions=None, name='seqs', name_qual='quals', include_insertions=False, return_quality=False, empty_chars=None, empty_quals=None, return_column_positions=False, min_ins_count=0, maintain_read_order=True, lowercase_insertions=True, ins_gap='-'):
        """
        positions (array/list): positions that we want to slice from table
        name (string): Name of the sequence column that is returned
        name_qual (string): Name of the quality column that is returned
        include_insertions (bool): If true, then positions containing insertions will also be reported

            .. note:: Returning column positions

                If a read does NOT have an insertion at that position it will be reported using the empty_chars/null value. 
                It is reccommended to also return_column_positions when returning insertions. 
                This will help to understand which positions in the sliced sequenes are insertions found in a subset of the data and which are non insertions in the reference.

                i.e.  READ1 = AACaTGT
                      READ2 = AACTGT
                      READ3 = AACTGA

                slicing sequences will appear as (if empty_chars=None or empty_chars='N')

                        AACATGT
                        AACNTGT
                        AACNTGA

        """
        if empty_chars is None:
            empty_chars = self.fill_na_val
                
        if positions is None:
            positions = self.position.values

        positions = OrderedSet(list(positions))
        
        # confirm that all positions are present in the column
        missing_pos = positions - OrderedSet(self.position.values)

        if len(missing_pos) > 0:            
            new_positions = list(positions & OrderedSet(self.position.values))
            prepend = empty_chars * (np.array(positions) < self.position.values.min()).sum() # ''.join([empty_chars for p in positions if p < self.position.values.min()])
            append =  empty_chars * (np.array(positions) > self.position.values.max()).sum() # ''.join([empty_chars for p in positions if p > self.position.values.max()])
            positions = new_positions            
            warnings.warn("The sequences do not cover all positions requested. {0}'s will be appended and prepended to sequences as necessary".format(empty_chars))
        else:
            prepend = ''
            append = ''

        if include_insertions and self.insertions is not None and self.insertions.empty is False:
            tmp_data = self.view_with_ins(positions, min_ins_count=min_ins_count, include_quality=return_quality, return_as_dataframe=False, lowercase_insertions=lowercase_insertions, ins_gap=ins_gap)
        else:
            tmp_data = self.loc[:, positions]
            # tmp_data.position = [(c, 0) for c in tmp_data.position.values]

        if maintain_read_order is True:
            # re-sort data
            tmp_data = tmp_data.loc[self.read.values]

        num_chars = tmp_data.shape[1]

        if num_chars == 0:
            # nothing to return
            if return_quality:
                qual_empty = self.null_qual * (len(prepend) + len(append))
                return pd.DataFrame({name: prepend + append, name_qual: qual_empty}, columns=[name, name_qual], index=tmp_data.read.values)
            else:
                return pd.DataFrame(prepend + append, columns=[name], index=tmp_data.read.values)

        # slice data into sequences
        if 'type' in tmp_data.dims:
            substring = pd.DataFrame(
                {
                    name: tmp_data.sel(type='seq').values.copy().view('S{0}'.format(num_chars)).ravel()
                }, index=tmp_data.read.values
            )
        else:
            substring = pd.DataFrame({
                name: tmp_data.values.copy().view('S{0}'.format(num_chars)).ravel()
            }, index=tmp_data.read.values)
            
        # if prepend or append:
            # substring['seqs'] = prepend + substring['seqs'] + append  # substring['seqs'].apply(lambda x: prepend + x + append)

        if 'type' in tmp_data.dims and return_quality is True:
            subquality = tmp_data.sel(type='quality').values.copy().view('S{0}'.format(num_chars)).ravel()
            substring[name_qual] = subquality            
            # if prepend or append:
            #     prepend = self.null_qual.encode() * len(prepend)
            #     append = self.null_qual.encode() * len(append)     
            #     print((prepend + substring['seqs'] + append))
            #     substring['quals2'] = prepend + substring['seqs'] + append  # .apply(lambda x: prepend + x + append)
            #     print(substring.head())
        # print(substring)

        substring = substring.applymap(lambda x: x.decode())

        if prepend or append:
            # add null values to front and end of sliced sequence
            substring[name] = prepend + substring[name] + append        
            # add in quality scores
            if 'type' in tmp_data.dims and return_quality is True:    
                if empty_quals is None:
                    empty_quals = self.null_qual
                prepend_qual = empty_quals * len(prepend)
                append_qual = empty_quals * len(append)            
                substring[name_qual] = prepend_qual + substring[name_qual] + append_qual

        if return_column_positions is True:
            return substring, tmp_data.position.values
        else:
            return substring  #.applymap(lambda x: x.decode())

    def subsample(self, numseqs, replace=False):
        """
            Return a random sample of sequences as a new object

            Args:
                numseqs (int): How many sequences to sample

            Returns:
                SeqTable Object
        """
        # random_sequences = self.seq_df.sample(numseqs)
        return self.isel(read=list(np.random.choice(self.shape[0], numseqs, replace=replace)))

    def _align_ref_seqs(self, reference_seqs, ref_seq_positions, reference_seq_ids=None):
        if isinstance(reference_seqs, string_types):
            reference_seqs = [reference_seqs]
        else:
            reference_seqs = list(reference_seqs)

        if reference_seq_ids is None:
            reference_seq_ids = np.arange(1, 1 + len(reference_seqs))
        else:
            assert len(reference_seq_ids) == len(reference_seqs), "The names associated with reference sequences must match the number of sequences provided"

        max_seq_len = max([len(s) for s in reference_seqs])
        # print(reference_seqs, [len(s) for s in reference_seqs], max_seq_len)

        # if ref_seq_positions is None:
        #     # assume that the position of the reference sequences are already lined up to the reference sequence
        #     ref_seq_positions = self.position.values[:max_seq_len]
        # use the maximum sequence length observed in list of sequences in seqtable to rmeove other reference positions that are irrelevent (past the region of interest)
        ref_seq_positions = list(ref_seq_positions)[:max_seq_len]

        # print(reference_seqs)
        return seqs_to_datarray(reference_seqs, pos=ref_seq_positions, index=np.array(list(reference_seq_ids)), seq_type=self.seq_type).sel(type='seq')

    @classmethod
    def _get_positions(cls, set_diff, p1, p2, positions_to_compare=None):
        if set_diff is False:
            if positions_to_compare is None:
                # only consider the intersection between reference_seqs and seqtable
                positions_to_compare = list(OrderedSet(list(p1)) & OrderedSet(list(p2)))
            else:
                # only consider the intersection between all positions
                positions_to_compare = list(OrderedSet(list(p1)) & OrderedSet(list(p2)) & OrderedSet(positions_to_compare))
        else:
            # change positions of interest to be the SET DIFFERENCE of positions parameter
            overlapping_positions = list(OrderedSet(list(p1)) & OrderedSet(list(p2)))
            if positions_to_compare is None:
                raise Exception('You cannot analyze the set-difference of all positions. Returns a non-informative answer (no columns to compare)')
            positions_to_compare = sorted(list(OrderedSet(overlapping_positions) - OrderedSet(positions_to_compare)))

        return positions_to_compare

    @classmethod
    def _sort_merged_columns(cls, x):
        return x if isinstance(x, tuple) else (x, 0)

    @classmethod
    def _make_positions_multiindex(cls, columns, names=[]):
        return pd.MultiIndex.from_tuples(
            [x if isinstance(x, tuple) else (x, 0) for x in columns],
            names=names
        )
        # return [
        #     [x[0] if isinstance(x, tuple) else x for x in columns],
        #     [x[1] if isinstance(x, tuple) else 0 for x in columns],
        # ]

    def _check_positions(self, positions):
        if not(positions is None):
            s1 = OrderedSet(list(self.position.values))
            s2 = OrderedSet(list(positions))
            if s1.issuperset(s2) is False:
                warnings.warn('Warning we cannot perform the request at all positions provided as they are undefined in this sequence table. \
                    The following positions will be ignored: {0}'.format(','.join([str(_) for _ in (s2 - s1)])))
            positions = [p for p in positions if p in s1]
        else:
            positions = list(self.position.values)

        return positions

    def compare_to_references(
            self, reference_seqs, positions_to_compare=None, ref_seq_positions=None, flip=False,
            set_diff=False, ignore_characters=[], treat_as_match=[], return_num_bases=False,
            names=None, reference_seq_ids=None, return_as_dataframe=False
    ):
        """
            Calculate which positions within a reference are not equal in all sequences in dataframe

            Args:
                reference_seqs (string or list of strings): A set of sequence(s) you want to compare the sequences to (insertions are not considered)
                positions_to_compare (list, default=None): specific positions in both the reference_seq and sequences you want to compare
                ref_seq_positions (int, default=0): positions_represented_by_refseq
                flip (bool): If True, then find bases that ARE MISMATCHES(NOT equal) to the reference
                set_diff (bool): If True, then we want to analyze positions that ARE NOT listed in positions parameters
                ignore_characters (char or list of chars): When performing distance/finding mismatches, always IGNORE THESE CHARACTERS, DONT TREAT THEM AS A MATCH OR A MISMATCH

                    ..important:: Change in datatype

                        If turned on, then the datatype returned will be of FLOAT and not BOOL. This is because we cannot represent np.nan as a bool, it will alwasy be treated as true

                treat_as_match (char or list of chars): When performing distance/finding mismatches, these BASES WILL ALWAYS BE TREATED AS TRUE/A MATCH
                return_num_bases (bool): If true, returns a second argument defining the number of relevant bases present in each row

                    ..important:: Change in output results

                        Setting return_num_bases to true will change how results are returned (two elements rather than one are returned)

            Returns:
                Dataframe of boolean variables showing whether base is equal to reference at each position
        """

        if names is None:
            names = ['', '']
            names[0] = self._name if hasattr(self, 'name') and self._name else 'Read_1'
            names[1] = 'Read_2'
        else:
            assert len(names) == 2
            if not names[0]:
                names[0] = self._name if hasattr(self, 'name') and self._name else 'Read_1'
            if not names[1]:
                names[1] = 'Read_2'

        if ref_seq_positions is None:
            # assume that the position of the reference sequences are already lined up to the reference sequence
            ref_seq_positions = self.position.values
        # print(ref_seq_positions)

        reference_seqs = self._align_ref_seqs(reference_seqs, ref_seq_positions, reference_seq_ids)

        # print(reference_seqs.position)
        positions_to_compare = self._get_positions(set_diff, self.position.values, reference_seqs.position.values, positions_to_compare)
        # print(positions_to_compare)
        res = numpy_ops.compare_sequence_matrices(
            self.loc[:, positions_to_compare, 'seq'].values, reference_seqs.loc[:, positions_to_compare].values, flip,
            treat_as_match, ignore_characters, return_num_bases
        )
        # print('res')
        # print(positions_to_compare)
        # print(res[0].shape, res.shape, self.read.values.shape, len(positions_to_compare), reference_seqs.read.values.shape)
        xrtmp = xr.DataArray(
            res[0] if return_num_bases is True else res, dims=(names[0], 'position', names[1]),
            coords={names[0]: self.read.values, 'position': positions_to_compare, names[1]: reference_seqs.read.values}
        )

        if return_as_dataframe:
            xrtmp = xrtmp.stack(z=(names[0], names[1])).T
            xrtmp = pd.DataFrame(
                xrtmp.values, columns=pd.Index(
                    xrtmp.position, name='position'
                ),
                index=pd.MultiIndex.from_tuples(xrtmp.z.values, names=names)
            )

        if return_num_bases is True:
            return xrtmp, res[1]
        else:
            return xrtmp

    def hamming_distance(
            self, reference_seqs, positions_to_compare=None, ref_seq_positions=None,
            set_diff=False, ignore_characters=[], treat_as_match=[], normalized=False,
            names=None, return_as_dataframe=True, reference_seq_ids=None
    ):
        """
            Determine hamming distance of all sequences in dataframe to a reference sequence.

            Args:
                reference_seqs (string): A string that you want to align sequences to
                positions_to_compare (list, default=None): specific positions in both the reference_seq and sequences you want to compare
                ref_seq_positions (int, default=0): where does the reference sequence start with respect to the aligned sequences
                set_diff (bool): If True, then we want to analyze positions that ARE NOT listed in positions parameters
                normalized (bool): If True, then divides hamming distance by the number of relevant bases
                names (list of strings): names to associate with table and reference sequences
                return_as_dataframe (bool, default=True): whether to return result as dataframe or xarray

            Returns:
                hamming_result (pd.DataFrame): NxM matrix of hamming distances for each sequence in sequence table and each sequence defined in reference sequences where N = # of sequences in sequence table and M = # of sequences in reference list
        """
        if normalized is True:
            diffs, bases = self.compare_to_references(
                reference_seqs, positions_to_compare, ref_seq_positions,
                flip=True, set_diff=set_diff, ignore_characters=ignore_characters, treat_as_match=treat_as_match,
                names=names, return_num_bases=True, return_as_dataframe=return_as_dataframe, reference_seq_ids=reference_seq_ids
            )
            hamming_result = ((diffs.sum(axis=1) / bases))
        else:
            hamming_result = self.compare_to_references(
                reference_seqs, positions_to_compare, ref_seq_positions,
                flip=True, set_diff=set_diff, ignore_characters=ignore_characters, treat_as_match=treat_as_match,
                names=names, return_as_dataframe=return_as_dataframe, reference_seq_ids=reference_seq_ids
            ).sum(axis=1)

        if return_as_dataframe:
            hamming_result = hamming_result.unstack()

        return hamming_result

    def calculate_pwm(self, pwm, positions=None, pwm_column_names='ACTG', null_scores=1):
        positions = self._check_positions(positions)
        return pd.DataFrame(
            numpy_ops.seq_pwm_ascii_map_and_score(pwm, self.loc[:, positions, 'seq'].values, pwm_column_names=pwm_column_names, use_log_before_sum=True, null_scores=null_scores),
            index=self.read.values,
            columns=['PWM_score']
        )

    def get_seq_dist(self, positions=None, method='counts', ignore_characters=[], weight_by=None, include_insertion_counts=False):
        """
            Returns the distribution of unique letter observed at each position in seqarray
        """

        allowed_methods = ['counts', 'freq', 'bits']
        assert method in allowed_methods, "invalid parameter passed in to method. Only allow the following: {0}".format(','.join(allowed_methods))

        if weight_by is not None:
            try:
                if isinstance(weight_by, pd.Series):
                    assert(weight_by.shape[0] == self.shape[0])
                    weight_by = weight_by.values
                elif isinstance(weight_by, pd.DataFrame):
                    assert(weight_by.shape[0] == self.shape[0])
                    assert(weight_by.shape[1] == 1)
                    weight_by = weight_by.values
                else:
                    assert(len(weight_by) == self.shape[0])
                    weight_by = np.array(weight_by)
            except:
                raise Exception('The provided weights for each sequence must match the number of input sequences!')

        if not(positions is None):
            s1 = OrderedSet(list(self.position.values))
            s2 = OrderedSet(list(positions))
            if s1.issuperset(s2) is False:
                warnings.warn('Warning we cannot provide sequence letter distribution at all positions provided as they are undefined in this sequence table. \
                    The following positions will be ignored: {0}'.format(','.join([str(_) for _ in (s2 - s1)])))
            positions = [p for p in positions if p in s1]
        else:
            positions = list(self.position.values)

        seq_dist_arr = self.loc[:, positions, 'seq'].values

        dist = numpy_ops.numpy_value_counts_bin_count(seq_dist_arr, weight_by)   # compare.apply(pd.value_counts).fillna(0)

        if include_insertion_counts:
            insertion_events = self.get_insertion_events(positions=positions, include_empty_positions=False, min_quality=0)
            dist = pd.concat([dist, pd.DataFrame(insertion_events.values, index=insertion_events.index, columns=[ord('^')]).T])

        dist.rename({c: chr(c) for c in list(dist.index)}, columns={i: c for i, c in enumerate(positions)}, inplace=True)

        drop_values = list(OrderedSet(ignore_characters) & OrderedSet(list(dist.index)))
        dist = dist.drop(drop_values, axis=0)

        if method == 'freq':
            dist = dist.astype(float) / dist.sum(axis=0)
        elif method == 'bits':
            N = self.shape[0]
            dist = get_bits(dist.astype(float) / dist.sum(axis=0), N, alphabet=list(dist.index))

        return dist.fillna(0)

    def get_substrings(self, word_length, positions=None, subsample_seqs=None, weights=None, include_insertions=False, min_ins_count=0, lowercase_insertions=True, ins_gap='-'):
        """
            Useful function for counting the occurrences of all possible SUBSTRINGs within a sequence table

            Lets say we have the following sequences:
            ACTW
            ATTA

            We want to get the occurrences of all combinations of substrings of length 3 in each sequence.
            For example
                1. we can have ACT, ACW, CTW, ATW in the first sequence
                2. we can have ATT, ATA, TTA, ATA in the second sequence

            Args:
                word_length (int): the length of substrings
                subsample_seqs (int): If provided, then will take only a random subsampling of the data before performing substring function

            Returns:
                dataframe: rows of dataframe are unique sequences of a given word length, Columns represents a specific combination of charcters in the word

                    .. note::Dataframe format

                        1. The number of columns should be equal to the total number of combinations (n choose k) where n = length of characters in seqtable, k = word_length
                        2. The sum of all rows in the dataframe should be equal to the the total number of sequences passed into the function

            Examples:
                >>> import seq_tables
                >>> st = seq_tables.SeqTable(['ACTW', 'ATTA'])
                >>> tmp = st.get_substrings(3)
                Returns:
                        (1, 2, 3) (1, 2, 4) (1, 3, 4) (2, 3, 4)
                    ACT    1          0         0         0
                    ACW    0          1         0         0
                    CTW    0          0         0         1
                    ATW    0          0         1         0
                    ATA    0          1         1         0
                    ATT    1          0         0         0

                    TTA    0          0         0         1

        """
        def dict_count(arr):
            dict_words = defaultdict(float)

            def arr_fxn(a, b):
                dict_words[a] += float(b)
                return 0

            assert(arr.shape[1] == 2)

            np.apply_along_axis(lambda x: arr_fxn(x[0], x[1]), arr=arr, axis=1)
            return dict_words

        def col_to_str(col):
            if isinstance(col, tuple):
                if col[1] == 0:
                    return 'p' + str(col[0])
                else:
                    return 'p{0}_ins_{1}'.format(col[0], abs(col[1]))
            else:
                return 'p' + str(col)

        if include_insertions and self.insertions is not None and self.insertions.empty is False:
            # stack insertions with bases
            tmp_table = self.view_with_ins(positions=positions, min_ins_count=min_ins_count, return_as_dataframe=False, include_quality=False, lowercase_insertions=lowercase_insertions, ins_gap=ins_gap)
        else:
            # only slice the columns without insertions
            tmp_table = self.loc[:, list(self.position.values) if positions is None else list(positions), 'seq']

        tmp_table = tmp_table if subsample_seqs is None else tmp_table.iloc[np.random.choice(self.shape[0], subsample_seqs)[0]]

        # convert column names to a string??
        pos_as_string = [col_to_str(p) for p in tmp_table.position.values]
        mapper = {c: i for i, c in enumerate(pos_as_string)}
        rev_mapper = {i: c for i, c in enumerate(pos_as_string)}
        substrings = [[mapper[x] for x in i] for i in itertools.combinations(list(pos_as_string), word_length)]
        table_values = tmp_table.values.view('S1')
        view_value = 'S' + str(word_length)

        dataframes = []

        if weights is None:
            for s in substrings:
                [a, b] = np.unique(table_values[:, s].reshape(-1).view(view_value), return_counts=True)
                dataframes.append(pd.DataFrame(b, index=a, columns=[tuple([rev_mapper[c] for c in s])]))
            substring_counts_df = pd.concat(dataframes, axis=1).fillna(0)
        else:
            for s in substrings:
                arr = table_values[:, s].reshape(-1).view(view_value)
                [a, b] = np.unique(arr, return_inverse=True)
                c = numpy_ops.numpy_value_counts_bin_count(b, weights=weights).reset_index().values
                a = a[c[:, 0].astype(int)]
                b = c[:, 1]
                dataframes.append(pd.DataFrame(b, index=a, columns=[tuple([rev_mapper[c] for c in s])]))

            substring_counts_df = pd.concat(dataframes, axis=1).fillna(0)
        if self.encoding_setting[0] is False:
            substring_counts_df.index = substring_counts_df.index.map(lambda x: x.decode())
        return substring_counts_df

    def get_insertion_seq_dist(self, positions=None, method='counts', min_ins_count=0):
        if positions is None:
            ins_df = self.insertions
        else:
            positions = list(self.insertions.index.levels[1].intersection(positions))
            ins_df = self.insertions.loc[(slice(None), positions), :]
        ins_dist = ins_df.reset_index().groupby(by=['position_ins', 'loc_ins']).seq.value_counts().unstack().fillna(0)
        total_ins = ins_dist.sum(axis=1)
        ins_dist = ins_dist[total_ins >= min_ins_count]
        ins_dist.rename(columns={c: chr(c) for c in ins_dist.columns}, inplace=True)
        ins_dist['-'] = self.shape[0] - ins_dist.sum(axis=1)
        ins_dist = ins_dist.T
        if method == 'freq':
            ins_dist = ins_dist.astype(float) / ins_dist.sum(axis=0)
        elif method == 'bits':
            N = self.shape[0]
            ins_dist = get_bits(ins_dist.astype(float) / ins_dist.sum(axis=0), N, alphabet=list(ins_dist.index))
        return ins_dist

    def mutation_profile(
        self, reference_seqs, positions_to_compare=None, ref_seq_positions=None,
        set_diff=False, ignore_characters=[], treat_as_match=[], normalized=False,
        reference_seq_ids=None, aggregate_positions=True
    ):
        """
            Return the type of mutation rates observed between the reference sequences and sequences in table.

            Args:
                reference_seqs (string): A string that you want to align sequences to
                positions_to_compare (list, default=None): specific positions in both the reference_seq and sequences you want to compare
                ref_seq_positions (int, default=0): where does the reference sequence start with respect to the aligned sequences
                set_diff (bool): If True, then we want to analyze positions that ARE NOT listed in positions parameters
                ignore_characters: (char or list of chars): When performing distance/finding mismatches, always IGNORE THESE CHARACTERS, DONT TREAT THEM AS A MATCH OR A MISMATCH
                normalized (bool): If True, then frequency of each mutation
                aggregate_positions (bool, True): If True then it will sum all mutation types observed at each position

            Returns:
                profile (pd.DataFrame): Returns the counts (or frequency) for each mutation observed (i.e. A->C or A->T)
        """
        def return_mutation_events(df):
            # dataframe should represent a matrix of the occurences of each base at a specific position to a reference.
            # the position at the reference is represented by the references base value at that position

            # stack the dataframe so that the reference base at a position is aligned with a possible representation of a read base at that position
            tmp = df.stack().reset_index()

            if normalized:
                tmp[0] = tmp[0] / (tmp[0].sum())

            # remove rows from stacked dataframe where reference sequence = read sequence OR where reference sequence should ingore a specific read sequence/treat it is a match
            tmp = tmp[(tmp['ref_seq'] != tmp['read_seq']) & (~tmp['read_seq'].isin(treat_as_match))]

            # group together all mutation events (i.e. A-> C, A-> T) regardless of position
            group_keys = ['ref_seq', 'read_seq'] if aggregate_positions else ['position', 'ref_seq', 'read_seq']
            return tmp.groupby(by=group_keys)[0].sum()

        if ref_seq_positions is None:
            # assume that the position of the reference sequences are already lined up to the reference sequence
            ref_seq_positions = self.position.values

        reference_seqs = self._align_ref_seqs(reference_seqs, ref_seq_positions)
        positions_to_compare = self._get_positions(set_diff, self.position.values, reference_seqs.position.values, positions_to_compare)

        # get a breakdown of the distribution of letteres at each position of interest
        dist = self.get_seq_dist(positions=positions_to_compare, method='counts', ignore_characters=ignore_characters)

        # create a dataframe where we define the expected letter of each base in each reference sequence
        dist_with_reference_letter = pd.DataFrame(
            np.tile(dist.values, (1, reference_seqs.shape[0])),  # repeat the distribution so that we can include it for every sequence provided to reference_seqs
            index=pd.Index(dist.index, name='read_seq'),
            columns=pd.MultiIndex.from_arrays(  # each column will be a multi index d
                [
                    np.repeat(reference_seq_ids if reference_seq_ids else np.arange(1, 1 + reference_seqs.shape[0]), len(positions_to_compare)),
                    np.tile(positions_to_compare, (reference_seqs.shape[0])),
                    reference_seqs.loc[:, positions_to_compare].values.ravel()
                ],
                names=['reference', 'position', 'ref_seq']
            )
        )

        mutation_df = dist_with_reference_letter.T.groupby(level=0).apply(return_mutation_events).unstack(level=[0])

        return mutation_df
        # return self.compare_to_reference(reference_seqs, positions_to_compare, ref_seq_positions, flip=True, treat_as_true=treat_as_true, set_diff=set_diff)

        # # def reference sequence
        # ref = pd.DataFrame(
        #     self.adjust_ref_seq(reference_seq, self.seq_table.columns, ref_start, return_as_np=True, positions=positions)[0],
        #     index=self.seq_table.columns
        # ).rename(columns={0: 'Ref base'}).transpose()
        # # compare all bases/residues to the reference seq (returns a dataframe of boolean vars)
        # not_equal_to = self.compare_to_reference(reference_seq, positions, ref_start, flip=True, treat_as_true=treat_as_true, set_diff=set_diff)
        # # now create a numpy array in which the reference is repeated N times where n = # sequences
        # ref = ref[not_equal_to.columns]
        # ref_matrix = np.tile(ref, (self.seq_table.shape[0], 1))
        # # now create a numpy array of ALL bases in the seq table that were not equal to the reference
        # subset = self.seq_table[not_equal_to.columns]
        # var_bases_unique = subset.values[(not_equal_to.values)]

        # # now create a corresponding numpy array of ALL bases in teh REF TABLE where that base was not equal in the seq table
        # # each index in this variable corresponds to the index (seq #, base position) in var_bases_unique
        # ref_bases_unique = ref_matrix[(not_equal_to.values)]

        # # OK lets do some fancy numpy methods and merge the two arrays, and then convert the 2D into 1D using bit conversion
        # # found this at: https://www.reddit.com/r/learnpython/comments/3v9y8u/how_can_i_find_unique_elements_along_one_axis_of/
        # mutation_combos = np.array([ref_bases_unique, var_bases_unique]).T.copy().view(np.int16)

        # # finally count the instances of each mutation we see (use squeeze(1) to ONLY squeeze single dim)
        # counts = np.bincount(mutation_combos.squeeze(1))
        # unique_mut = np.nonzero(counts)[0]

        # counts = counts[unique_mut]
        # # convert values back to chacters of format (REF BASE/RESIDUE, VAR base/residue)
        # unique_mut = unique_mut.astype(np.uint16).view(np.uint8).reshape(-1, 2).view('S1').astype('U1')

        # # unique_mut, counts = np.unique(mutation_combos.squeeze(), return_counts=True) => this could have worked also, little slower
        # if len(unique_mut) == 0:
        #     return pd.Series()
        # mut_index = pd.MultiIndex.from_tuples(list(unique_mut), names=['ref', 'mut'])
        # mutation_counts = pd.Series(index=mut_index, data=counts).astype(float).sort_index()

        # del ref_bases_unique
        # del var_bases_unique
        # del mutation_combos

        # if ignore_characters:
        #     # drop any of these mutation types => maybe i dont need to remove from axis of 0 (the provided reference)??
        #     mutation_counts = mutation_counts.unstack().drop(ignore_characters, axis=1, errors='ignore').drop(ignore_characters, axis=0, errors='ignore').stack()

        # if normalized is True:
        #     mutation_counts = mutation_counts / (mutation_counts.sum())

        # return mutation_counts

    def mutation_TS_TV_profile(
        self, reference_seqs, positions_to_compare=None, ref_seq_positions=None,
        set_diff=False, ignore_characters=[], treat_as_match=[], normalized=False,
        reference_seq_ids=None, aggregate_positions=True
    ):
        """
            Return the ratio of transition rates (A->G, C->T) to transversion rates (A->T/C) observed between the reference sequence and sequences in table.

            Args:
                reference_seq (string): A string that you want to align sequences to
                positions (list, default=None): specific positions in both the reference_seq and sequences you want to compare
                ref_start (int, default=0): where does the reference sequence start with respect to the aligned sequences
                set_diff (bool): If True, then we want to analyze positions that ARE NOT listed in positions parameters

            Returns:
                ratio (float): TS Freq / TV Freq
                TS (float): TS Freq
                TV (float): TV Freq
        """

        if self.seq_type != 'NT':
            raise('Error: you cannot calculate TS and TV mutations on AA sequences. Either the seq_type is incorrect or you want to use the function mutation_profile')
        transitions = [('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')]
        transversions = [
            ('A', 'C'), ('C', 'A'), ('A', 'T'), ('T', 'A'),
            ('G', 'C'), ('C', 'G'), ('G', 'T'), ('T', 'G'),
        ]

        def pos_agg_fxn(df, superset):
            type_found = list(set(df.index.values) - set(superset))
            return df.loc[type_found].sum() / df.sum()

        mutations = self.mutation_profile(
            reference_seqs, positions_to_compare, ref_seq_positions,
            set_diff, ignore_characters, reference_seq_ids=reference_seq_ids, aggregate_positions=aggregate_positions
        )
        if mutations.shape[0] == 0:
            return np.nan, 0, 0

        transitions_found = list(set(mutations.index.values) - set(transitions))
        transversions_found = list(set(mutations.index.values) - set(transversions))

        mutations = mutations.reset_index().set_index(['ref_seq', 'read_seq'])

        if aggregate_positions is True:
            ts_freq = mutations.loc[transitions_found].sum() / mutations.sum()
            tv_freq = mutations.loc[transversions_found].sum() / mutations.sum()
        else:
            ts_freq = mutations.groupby(by=['position']).apply(lambda x: pos_agg_fxn(x, transitions)).drop('position', axis=1, errors='ignore')
            tv_freq = mutations.groupby(by=['position']).apply(lambda x: pos_agg_fxn(x, transversions)).drop('position', axis=1, errors='ignore')

        return ts_freq / tv_freq, ts_freq, tv_freq

    def quality_filter(self, q, p, ignore_null_qual=True):
        """
            Filter out sequences based on their average qualities at each base/position

            ..note:: positions

                Insertion qualities are currently ignored for these sequence reads

            Args:
                q (int): quality score cutoff
                p (int/float/percent 0-100): the percent of bases that must have a quality >= the cutoff q
                inplace (boolean): If False, returns a copy of the object filtered by quality score
                ignore_null_qual (boolean): Ignore bases that are not represented. (i.e. those with quality of 0)
        """
        if 'quality' not in self.type.values:
            raise Exception("You have not passed in any quality data for these sequences")

        quality_scores = self.sel(type='quality').values.view(np.uint8) - self.phred_adjust
        total_bases = (quality_scores > (ord(self.null_qual) - self.phred_adjust)).sum(axis=1) if ignore_null_qual else self.shape[1]
        percent_above = (100.0 * ((quality_scores >= q).sum(axis=1))) / (1.0 * total_bases)

        filtered = SeqTable(self[percent_above >= p])
        filtered.attrs = copy.deepcopy(self.attrs)
        ins_df = filtered.attrs.get('seqtable', {}).get('insertions').copy()
        if not(ins_df is None) and ins_df.empty is False:
            # the following reads are still presnt
            present_reads = list(ins_df.index.levels[0].intersection(filtered.read.values))
            # now filter out the reads
            filtered.attrs['seqtable']['insertions'] = ins_df.loc[present_reads].copy()
            
            # filtered.insertions = filtered.attrs['seqtable']['insertions']
            del ins_df
        else:
            filtered.attrs['seqtable']['insertions'] = pd.DataFrame()

        return filtered

    def convert_low_bases_to_null(self, q, replace_with=None, inplace=False, remove_from_insertions=True, ignore_null_qual=True):
        """
            This will convert all letters whose corresponding quality is below a cutoff to the value replace_with

            Args:
                q (int): quality score cutoff, convert all bases whose quality is < than q
                inplace (boolean): If False, returns a copy of the object filtered by quality score
                replace_with (char): a character to replace low bases with

                    ..Note:: None

                        When replace with is set to None, then it will replace bases below the quality with the objects fill_na attribute

                ignore_null_qual (boolean): When true then it will NOT convert any bases whose base quality is 0
        """

        ignore_qual = -1 if ignore_null_qual is False else 0

        if 'quality' not in self.type.values:
            raise Exception("You have not passed in any quality data for these sequences")

        if inplace is True:
            meself = self 
        else:
            meself = self.copy()
            meself.attrs = copy.deepcopy(self.attrs)

        if replace_with is None:
            replace_with = self.fill_na_val

        qual_as_num = meself.sel(type='quality').values.view(np.uint8) - self.phred_adjust

        meself.sel(type='seq').values[
            (qual_as_num < q) & (qual_as_num > ignore_qual)
        ] = replace_with

        if remove_from_insertions and 'insertions' in meself.attrs.get('seqtable', {}):
            ins_df = meself.attrs.get('seqtable', {}).get('insertions')
            ins_df = ins_df[(ins_df['quality'].values >= q) | (ins_df['quality'].values <= ignore_qual)]
            #meself.insertions = ins_df
            meself.attrs['seqtable']['insertions'] = ins_df

        if inplace is False:
            return meself

    def get_plogo(self, background_seqs=None, positions=None, background_seq_positions=None, ignore_characters=[], alpha=0.01):
        counts = self.get_seq_dist(positions, ignore_characters=ignore_characters)
        if background_seqs is not None:
            bkst = self._align_ref_seqs(background_seqs, background_seq_positions)
            bkst_freq = bkst.get_seq_dist(positions, method='freq', ignore_characters=ignore_characters)
        else:
            bkst_freq = None

        return get_plogo(counts, self.seq_type, bkst_freq, alpha=alpha)

    def pos_entropy(self, positions=None, ignore_characters=[], nbit=2):
        dist = self.get_seq_dist(positions, method='freq', ignore_characters=ignore_characters)
        return shannon_info(dist, nbit)

    def relative_entropy(self, background_seqs=None, positions=None, background_seq_positions=None, ignore_characters=[]):
        dist = self.get_seq_dist(positions, method='freq', ignore_characters=ignore_characters)
        if background_seqs is not None:
            bkst = self._align_ref_seqs(background_seqs, background_seq_positions)
            bkst_freq = bkst.get_seq_dist(positions, method='freq', ignore_characters=ignore_characters)
        else:
            bkst_freq = None
        return relative_entropy(dist, self.seq_type, bkst_freq)

    def get_quality_dist(self, positions=None, bins='even', exclude_null_quality=True, sample=None, percentiles=[10, 25, 50, 75, 90], stats=['mean', 'median', 'max', 'min'], plotly_sampledata_size=20, use_multiindex=True):
        """
            Returns the distribution of quality across the given sequence, similar to FASTQC quality seq report.

            Args:
                bins(list of ints or tuples, or 'fastqc', or 'even'): bins defines how to group together the columns/sequence positions when aggregating the statistics.

                    .. note:: bins='fastqc' or 'even'

                        if bins is not a set of numbers and instead one of the two predefined strings ('fastqc' and 'even') then calculation of bins will be defined as follows:

                                1. fastqc: Identical to the bin ranges used by fastqc report
                                2. even: Creates 10 evenly sized bins based on sequence lengths

                percentiles (list of floats, default=[10, 25, 50, 75, 90]): value passed into numpy percentiles function.
                exclude_null_quality (boolean, default=True): do not include quality scores of 0 in the distribution
                sample (int, default=None): If defined, then we will only calculate the distribution on a random set of subsampled sequences

            Returns:
                data (DataFrame): contains the distribution information at every bin (min value, max value, desired precentages and quartiles)
                graphs (plotly object): contains plotly graph objects for generating plots of the data afterwards

            Examples:
                Show the median of the quality at the first ten positions in the sequence

                >>> table = SeqTable(['AAAAAAAAAA', 'AAAAAAAAAC', 'CCCCCCCCCC'], qualitydata=['6AA9-C9--6C', '6AA!1C9BA6C', '6AA!!C9!-6C'])
                >>> box_data, graphs = table.get_quality_dist(bins=range(10), percentiles=[0.5])

                Now repeat the example from above, except group together all values from the first 5 bases and the next 5 bases
                i.e.  All qualities between positions 0-4 will be grouped together before performing median, and all qualities between 5-9 will be grouped together). Also, return the bottom 10 and upper 90 percentiles in the statsitics

                >>> box_data, graphs = table.get_quality_dist(bins=[(0,4), (5,9)], percentiles=[0.1, 0.5, 0.9])

                We can also plot the results as a series of boxplots using plotly
                >>> from plotly.offline import init_notebook_mode, iplot, plot, iplot_mpl
                # assuming ipython..
                >>> init_notebook_mode()
                >>> plotly.iplot(graphs)
                # using outside of ipython
                >>> plotly.plot(graphs)
        """
        assert 'quality' in self.type.values

        if positions is None:
            positions = list(self.position.values)

        return numpy_ops.get_quality_dist(
            self.loc[:, positions, 'quality'].values.view(np.uint8) - self.phred_adjust, positions,
            bins, exclude_null_quality, sample, percentiles, stats, plotly_sampledata_size, use_multiindex
        )

    def seq_logo(self, positions=None, include_insertions=True, weights=None, method='freq', ignore_characters=[], min_ins_count=0, **kwargs):
        dist_no_ins = self.get_seq_dist(positions, method, ignore_characters, weights)
        if include_insertions is True and (self.insertions is not None and self.insertions.shape[0] > 0):
            dist_with_ins = self.get_insertion_seq_dist(positions, method=method, min_ins_count=min_ins_count)
            merged_dist = pd.concat([dist_no_ins, dist_with_ins], axis=1).fillna(0)
        else:
            merged_dist = dist_no_ins.fillna(0)

        cols = merged_dist.columns
        sorted_column_indicies = sorted(
            range(len(cols)),
            key=lambda k: self._sort_merged_columns(cols[k])
        )

        merged_dist = merged_dist.iloc[:, sorted_column_indicies]
        return draw_seqlogo_barplots(merged_dist, alphabet=self.seq_type, **kwargs)

    def _get_filtered_insertions_by_quality(self, min_quality):
        if self.insertions is None:
            return pd.DataFrame()
        elif min_quality == 0:
            return self.insertions
        return self.insertions[self.insertions['quality'] >= min_quality].copy()

    def get_insertion_events(self, positions=None, include_empty_positions=False, min_quality=0):
        """
            Return the number of times an insertion occurs (at least once) at a specific position
        """
        ins_df = self._get_filtered_insertions_by_quality(min_quality)

        if ins_df.shape[0] > 0:
            if include_empty_positions:
                insertion_events = pd.Series(ins_df.loc[(slice(None), slice(None), -1), :].groupby(level=1).apply(len), index=positions, name='position')
            else:
                insertion_events = pd.Series(ins_df.loc[(slice(None), slice(None), -1), :].groupby(level=1).apply(len), name='position')
        else:
            insertion_events = pd.Series([np.nan])

        if not(positions is None):
            return insertion_events.loc[insertion_events.index.intersection(positions)]
        else:
            return insertion_events

    def get_insertion_distribution(self, positions=None, include_empty_positions=False, min_quality=0):
        ins_df = self._get_filtered_insertions_by_quality(min_quality).groupby(level=[1, 2]).apply(len).reset_index('loc_ins').rename(columns={0: 'counts'})

        if include_empty_positions:
            missing_indexes = OrderedSet(positions) - OrderedSet(ins_df.index)
            ins_df = pd.concat([ins_df, pd.DataFrame([np.nan] * len(missing_indexes), columns=['counts'], index=missing_indexes)]).sort_index()
            ins_df.index.name = 'position'
        else:
            ins_df.index.name = 'position'

        if not(positions is None):
            return ins_df.loc[ins_df.index.intersection(positions)]
        else:
            return ins_df

    def get_average_insertion_quality(self, positions=None, include_empty_positions=False):
        ins_df = self.insertions.groupby(level=[1, 2]).quality.apply(np.mean).reset_index('loc_ins')

        if include_empty_positions:
            missing_indexes = OrderedSet(positions) - OrderedSet(ins_df.index)
            ins_df = pd.concat([ins_df, pd.DataFrame([np.nan] * len(missing_indexes), columns=['quality'], index=missing_indexes)]).sort_index()
            ins_df.index.name = 'position'
        else:
            ins_df.index.name = 'position'

        if not(positions is None):
            return ins_df.loc[ins_df.index.intersection(positions)]
        else:
            return ins_df

    def get_insertion_expectations(self, positions=None, include_empty_positions=False, method='mean', min_quality=0):
        """
            Return an aggregated value that illustrates the type of insertion observed at each position (i.e. mean number of insertions, max number of insertions, min number of insertions, etc)
        """
        allowed_stats = ['mean', 'median', 'max']
        assert method in allowed_stats, "We only offer insertion statistics for the following: {0}".format(','.join(allowed_stats))

        ins_df = self._get_filtered_insertions_by_quality(min_quality)

        if ins_df.shape[0] > 0:
            # calculate the total number of times we have insertion data at each position
            total_p = ins_df.groupby(level=[1]).apply(len)

            tmp_agg = ins_df.groupby(level=[1, 2]).apply(len).reset_index('loc_ins')
            if method == 'mean':
                # return the expected # of insertions at that position
                # we calculate this as the weighted average where weight = # of times (groupby(level=0).sum()) we observed a specific insertion type at a position (pos 88, loc_ins -1)
                series = tmp_agg.product(axis=1).groupby(level=0).sum() / total_p
            elif method == 'max':
                # calculate the maximum insertion event even observed at a specific position (remember insertion events are labeled as negative so max = min...yea...)
                series = tmp_agg.groupby(level=0).loc_ins.min()
            elif method == 'median':
                series = tmp_agg.groupby(level=0).apply(lambda x: np.median(np.repeat(*(x.values.T))))
        else:
            series = [np.nan]

        if include_empty_positions:
            insertion_events = pd.Series(series, index=positions, name='position')
        else:
            insertion_events = pd.Series(series, name='position')
        if not(positions is None):
            return insertion_events.loc[insertion_events.index.intersection(positions)]
        else:
            return insertion_events

    # def get_consensus_depr(self, positions=None, modecutoff=0.5):
    #     """
    #         Returns the sequence consensus of the bases at the defined positions

    #         Args:
    #             positions: Slice which positions in the table should be conidered
    #             modecutoff: Only report the consensus base of letters which appear more than the provided modecutoff (in other words, the mode must be greater than this frequency)

    #         .. note:: deprecation

    #             This function appears to be slightly slower than the newer get_consensus function
    #     """
    #     compare = self.loc[:, positions, 'seq'].values.view(np.uint8) if positions else self.loc[:, :, 'seq'].values.view(np.uint8)
    #     cutoff = float(compare.shape[0]) * modecutoff
    #     chars = compare.shape[1]
    #     # dist = np.int8(compare.apply(lambda x: x.mode()).values[0])
    #     # dist = np.int8(compare.reduce(func=lambda x, axis: scipy.stats.mode(x), dim='position'))   # apply(lambda x: x.mode()).values[0])
    #     dist = np.apply_along_axis(arr=compare, func1d=lambda x: scipy.stats.mode(x), axis=0)
    #     # dist[0][0] # => mode
    #     # dist[1][1] # => counts for mode
    #     dist[0][0][dist[1][0] <= cutoff] = ord('N')

    #     seq = (np.uint8(dist[0][0])).view('S' + str(chars))[0]
    #     return seq

    def get_consensus(self, positions=None, modecutoff=0.5, include_insertions=True, return_column_positions=False, exclude_insertions_with_gap_cons=True):
        """
            Returns the sequence consensus of the bases at the defined positions

            Args:
                positions: Slice which positions in the table should be conidered
                modecutoff: Only report the consensus base of letters which appear more than the provided modecutoff (in other words, the mode must be greater than this frequency)
        """

        non_ins_dist = self.get_seq_dist(positions=positions)
        if include_insertions:
            ins_dist = self.get_insertion_seq_dist(positions=positions)
            merged_dist = pd.concat([non_ins_dist, ins_dist], axis=1).fillna(0)
        else:
            merged_dist = non_ins_dist.fillna(0)
        cols = merged_dist.columns
        sorted_column_indicies = sorted(
            range(len(cols)),
            key=lambda k: self._sort_merged_columns(cols[k])
        )

        merged_dist = merged_dist.iloc[:, sorted_column_indicies]
        below_cutoff = merged_dist.max(axis=0) <= merged_dist.sum(axis=0) * modecutoff
        cons_bases = np.array(merged_dist.index.values, dtype='U1')[merged_dist.values.argmax(axis=0)]  # => use this rather than idx max because (a) its faster, and (b) it returns a numpy array with data type S1 rathr than object
        cons_bases[below_cutoff] = 'N'

        if include_insertions is True and exclude_insertions_with_gap_cons is True:
            # lets figure out which are our insertion columns
            idx_ins_cols = [True if isinstance(c, tuple) else False for i, c in enumerate(merged_dist.columns)]
            # figure out which of the consensus bases are considered a gap
            cons_is_del = [c == '-' for c in cons_bases]
            # we only want to isolate either (1) not insertion positions, or (2) insertions positions tat do NOT have  gap
            dont_want = np.array(idx_ins_cols) & np.array(cons_is_del)
            keep_cols = ~dont_want
            cons_bases = cons_bases[keep_cols]
            cols = merged_dist.columns[keep_cols]

        seq = (cons_bases).view('U' + str(len(cols)))[0]
        if return_column_positions is True:
            return seq, cols
        else:
            return seq
