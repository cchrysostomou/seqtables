from __future__ import absolute_import
# from ..internals import xarray_extensions
import xarray as xr
from .internals import _seq_df_to_datarray, _seqs_to_datarray
from .seq_logo import draw_seqlogo_barplots, get_bits, get_plogo, shannon_info, relative_entropy
from . import numpy_ops
import warnings
import numpy as np
import pandas as pd
from six import string_types


def df_to_dataarray(df, seq_type, index, user_attrs={}, ref_name='', ref_to_pos_dict={}):
    """
        Converts a dataframe into an XARRAY object (not a seqtable)
    """
    ignore_ref_col = True if ref_name else False
    arrs, attrs = _seq_df_to_datarray(df, seq_type, index, ignore_ref_col=ignore_ref_col, ref_name=ref_name, ref_to_pos_dict=ref_to_pos_dict)
    attrs['user_defined'] = user_attrs
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


def from_df(*args, **kwargs):
    """
    convert a pandas dataframe containing sequences, quality scores, and or cigar strings into a seq tables object
    """
    new_st = SeqTable(df_to_dataarray(*args, **kwargs))
    new_st.update_attributes()
    return new_st


def from_list(*args, **kwargs):
    new_st = SeqTable(seqs_to_datarray(*args, **kwargs))
    new_st.update_attributes()
    return new_st


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

    phred_adjust = 33
    fill_na_val = 'N'
    null_qual = '!'
    insertions = None
    seq_type = ''

    def __init__(self, seq_list, *args, **kwargs):
        if isinstance(seq_list, list) or isinstance(seq_list, np.ndarray) or isinstance(seq_list, pd.Series):
            xarr = seqs_to_datarray(seq_list, *args, **kwargs)
            super(SeqTable, self).__init__(xarr)  # data_vars=arrs, attrs=attrs)
            self.update_attributes()
        else:
            super(SeqTable, self).__init__(seq_list, *args, **kwargs)

    def update_attributes(self):
        self.phred_adjust = self.attrs['seqtable'].get('phred_adjust', 33)
        self.fill_na_val = self.attrs['seqtable'].get('fill_na', 'N')
        self.insertions = self.attrs['seqtable'].get('insertions')
        self.seq_type = self.attrs['seqtable'].get('seq_type')

    def get_sequences(self):
        return xr.DataArray(self.sel(type='seq'))

    def get_quality(self, as_num=True):
        if as_num:
            return xr.DataArray(
                self.sel(type='quality').values.view(np.uint8) - self.phred_adjust,
                dims=('read', 'position'),
                coords={'read': self.read, 'position': self.position}
            )
        else:
            return xr.DataArray(self.sel(type='quality'))

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

    @classmethod
    def _align_ref_seqs(cls, reference_seqs, ref_seq_positions, reference_seq_ids=None):
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
        ref_seq_positions = ref_seq_positions[:max_seq_len]

        # print(reference_seqs)
        return seqs_to_datarray(reference_seqs)

    @classmethod
    def _get_positions(cls, set_diff, p1, p2, positions_to_compare=None):
        if set_diff is False:
            if positions_to_compare is None:
                # only consider the intersection between reference_seqs and seqtable
                positions_to_compare = list(set(list(p1)) & set(list(p2)))
            else:
                # only consider the intersection between all positions
                positions_to_compare = list(set(list(p1)) & set(list(p2)) & set(positions_to_compare))
        else:
            # change positions of interest to be the SET DIFFERENCE of positions parameter
            overlapping_positions = list(set(list(p1)) & set(list(p2)))
            if positions_to_compare is None:
                raise Exception('You cannot analyze the set-difference of all positions. Returns a non-informative answer (no columns to compare)')
            positions_to_compare = sorted(list(set(overlapping_positions) - set(positions_to_compare)))

        return positions_to_compare

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

                treat_as_true (char or list of chars): When performing distance/finding mismatches, these BASES WILL ALWAYS BE TREATED AS TRUE/A MATCH
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

        reference_seqs = self._align_ref_seqs(reference_seqs, ref_seq_positions, reference_seq_ids)

        positions_to_compare = self._get_positions(set_diff, self.position.values, reference_seqs.position.values, positions_to_compare)

        res = numpy_ops.compare_sequence_matrices(
            self.loc[:, positions_to_compare, 'seq'].values, reference_seqs.loc[:, positions_to_compare].values, flip,
            treat_as_match, ignore_characters, return_num_bases
        )

        xrtmp = xr.DataArray(
            res[0] if return_num_bases is True else res, dims=(names[0], 'position', names[1]),
            coords={names[0]: self.read.values, 'position': positions_to_compare, names[1]: reference_seqs.read.values}
        )

        if return_as_dataframe:
            xrtmp = xrtmp.stack(z=(names[0], names[1])).T
            xrtmp = pd.DataFrame(xrtmp.values, columns=pd.Index(xrtmp.position, name='position'), index=pd.MultiIndex.from_tuples(xrtmp.z.values, names=names))

        if return_num_bases is True:
            return xrtmp, res[1]
        else:
            return xrtmp

    def hamming_distance(
            self, reference_seqs, positions_to_compare=None, ref_seq_positions=None,
            set_diff=False, ignore_characters=[], treat_as_match=[], normalized=False,
            names=None, return_as_dataframe=True
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
                flip=True, set_diff=set_diff, ignore_characters=ignore_characters,
                names=names, return_num_bases=True, return_as_dataframe=return_as_dataframe,
            )
            hamming_result = ((diffs.sum(axis=1) / bases))
        else:
            hamming_result = self.compare_to_references(
                reference_seqs, positions_to_compare, ref_seq_positions,
                flip=True, set_diff=set_diff, ignore_characters=ignore_characters,
                names=names, return_as_dataframe=return_as_dataframe
            ).sum(axis=1)

        if return_as_dataframe:
            hamming_result = hamming_result.unstack()

        return hamming_result

    def get_seq_dist(self, positions=None, method='counts', ignore_characters=[], weight_by=None):
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
            s1 = set(list(self.position.values))
            s2 = set(list(positions))
            if s1.issuperset(s2) is False:
                warnings.warn('Warning we cannot provide sequence letter distribution at all positions provided as they are undefined in this sequence table. \
                    The following positions will be ignored: {0}'.format(','.join([str(_) for _ in (s2 - s1)])))
            positions = [p for p in positions if p in s1]
        else:
            positions = list(self.position.values)

        seq_dist_arr = self.loc[:, positions, 'seq'].values

        dist = numpy_ops.numpy_value_counts_bin_count(seq_dist_arr, weight_by)   # compare.apply(pd.value_counts).fillna(0)

        dist.rename({c: chr(c) for c in list(dist.index)}, columns={i: c for i, c in enumerate(positions)}, inplace=True)

        drop_values = list(set(ignore_characters) & set(list(dist.index)))
        dist = dist.drop(drop_values, axis=0)

        if method == 'freq':
            dist = dist.astype(float) / dist.sum(axis=0)
        elif method == 'bits':
            N = self.shape[0]
            dist = get_bits(dist.astype(float) / dist.sum(axis=0), N, alphabet=list(dist.index))

        return dist.fillna(0)

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

        filtered = self[percent_above >= p]
        ins_df = filtered.attrs.get('seqtable', {}).get('insertions')
        if not(ins_df is None):
            # the following reads are still presnt
            present_reads = list(ins_df.index.levels[0].intersection(filtered.read.values))
            # now filter out the reads
            filtered.attrs['seqtable']['insertions'] = ins_df.loc[present_reads]
            filtered.insertions = filtered.attrs['seqtable']['insertions']
            del ins_df

        return filtered
        # if inplace is False:

        # else:
        #     self = self[percent_above >= p]

    def convert_low_bases_to_null(self, q, replace_with=None, inplace=False, remove_from_insertions=True):
        """
            This will convert all letters whose corresponding quality is below a cutoff to the value replace_with

            Args:
                q (int): quality score cutoff, convert all bases whose quality is < than q
                inplace (boolean): If False, returns a copy of the object filtered by quality score
                replace_with (char): a character to replace low bases with

                    ..Note:: None

                        When replace with is set to None, then it will replace bases below the quality with the objects fill_na attribute
        """
        if 'quality' not in self.type.values:
            raise Exception("You have not passed in any quality data for these sequences")

        meself = self if inplace is True else self.copy()

        if replace_with is None:
            replace_with = self.fill_na_val

        meself.sel(type='seq').values[meself.sel(type='quality').values.view(np.uint8) - self.phred_adjust < q] = replace_with

        if remove_from_insertions and 'insertions' in meself.attrs.get('seqtable', {}):
            ins_df = meself.attrs.get('seqtable', {}).get('insertions')
            ins_df = ins_df[ins_df['quality'].values >= q]
            meself.insertions = ins_df
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

    def get_quality_dist(self, positions=None, bins='fastqc', exclude_null_quality=True, sample=None, percentiles=[10, 25, 50, 75, 90], stats=['mean', 'median', 'max', 'min'], plotly_sampledata_size=20, use_multiindex=True):
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

        return numpy_ops.get_quality_dist(
            self.sel(type='quality').values.view(np.uint8) - self.phred_adjust, self.position.values,
            bins, exclude_null_quality, sample, percentiles, stats, plotly_sampledata_size, use_multiindex
        )

    def seq_logo(self, positions=None, weights=None, method='freq', ignore_characters=[], **kwargs):
        dist = self.get_seq_dist(positions, method, ignore_characters, weights)
        return draw_seqlogo_barplots(dist, alphabet=self.seq_type, **kwargs)

    def get_consensus(self, positions=None, modecutoff=0.5):
        """
            Returns the sequence consensus of the bases at the defined positions

            Args:
                positions: Slice which positions in the table should be conidered
                modecutoff: Only report the consensus base of letters which appear more than the provided modecutoff (in other words, the mode must be greater than this frequency)
        """
        compare = self.seq_table.loc[:, positions] if positions else self.seq_table
        cutoff = float(compare.shape[0]) * modecutoff
        chars = compare.shape[1]
        dist = np.int8(compare.apply(lambda x: x.mode()).values[0])
        dist[(compare.values == dist).sum(axis=0) <= cutoff] = ord('N')
        seq = dist.view('S' + str(chars))[0]
        return seq

