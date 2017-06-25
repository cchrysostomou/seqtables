"""
Using xarray to represent aligned sequences as a set of multidimensional tables

"""

from __future__ import absolute_import

import gc
import copy
import warnings
import pandas as pd
import math
import numpy as np
import itertools
from collections import defaultdict
import xarray as xr

# from collections import defaultdict
from .seq_logo import draw_seqlogo_barplots, get_bits, get_plogo, shannon_info, relative_entropy
from .seq_table_util import get_quality_dist, dna_alphabet, aa_alphabet  # , degen_to_base, dna_alphabet, aa_alphabet

from .utils.operations import strseries_to_bytearray
from .utils.unique_ops import numpy_value_counts_bin_count


class seqtable():
    """
    Class for viewing aligned sets of DNA or AA sequences. This will take in a list of sequences that are presumed to be aligned to one another, and
    convert the sequences into an xarray dataset (sets of numpy arrays). In additiona to the sequence, quality scores for each base can be provided in addition to sequences.

    The xarray dataset will be structured as follows:
        xr.DataSet({
            'reference_table': Represents a data-array of bases in each that reads that contains (deletions are allowed) to the reference sequence of interest
            'insertions': Represents a data-array of bases in each read that only includes insertions for each read with respect to the reference sequence of interest
        })


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
        >>> sq = seq_tables.seqtable(['AAA', 'ACT', 'ACA'])
        >>> sq.hamming_distance('AAA')
        >>> sq = read_fastq('fastqfile.fq')
    """
    def __init__(
        self, seq_data=None, quality_data=None, seq_position_values=1, index=None,
        seq_type=None, phred_adjust=33, null_qual='!', encode_letters=True  # , encoding='utf-8'  # , **kwargs
    ):
        assert seq_data is not None, "The default seqdata value is not allowed, you must provide sequences"
        self.null_qual = null_qual
        if seq_type is None:
            # attempt to guess the sequence type
            check_sequences = min(len(seq_data), 1000)
            if isinstance(seq_data, list):
                sub_seq = ''.join(seq_data[:check_sequences])
            else:
                try:
                    sub_seq = ''.join(list(seq_data.iloc[:check_sequences]))
                except:
                    raise Exception("Error: cannot determine seq_type given the provided seqdata. Either manually define the sequence type or provide seq data as a list")
            unique_letters = set(list(sub_seq.lower()))
            if len(unique_letters - set(list('actgn'))) == 0:
                # the first set of sequences had only actgn letters so we assume its a nucleotide sequence
                self.seq_type = 'NT'
            elif len(unique_letters - set(dna_alphabet)) == 0:
                # the first set of sequences have ACTGN letters and also letters that could refer to DEGENERATE bases. Or they could be a constricted set of AA residues so we arent sure, but default to nucleotide
                warnings.warn('The provided sequences appear to only have letters pertaining to DNA and degenerage DNA bases. So we are assuming that the provided sequences are DNA sequences. If this is incorrect, please manually define the seq_type when initializing the sequence table or change its sequence type via `change_seq_type` function')
                self.seq_type == 'NT'
            elif len(unique_letters - set(aa_alphabet)) == 0:
                # the first set of sequences have letters that only correspond to AA residues
                self.seq_type = 'AA'
            else:
                # the first set of sequences have letters that arent part of both DNA and AA values, so default to AA
                warnings.warn('The provided sequences appear to have letters outside the known/expected AA and NT nomenclature. The sequence type will be defaulted to AA sequences. If this is incorrect, please manually define the seq_type when initializing the sequence table or change its sequence type via `change_seq_type` function')
                self.seq_type = 'AA'
        elif seq_type not in ['AA', 'NT']:
            raise Exception('You defined seq_type as, {0}. We only allow seq_type to be "AA" or "NT"'.format(seq_type))
        else:
            # user appropriately defined the type of sequences in the list
            self.seq_type = seq_type

        # subtract ascii values of quality scores by this # (for example when phred_adjust = 33, then ! means quality score of 0)
        self.phred_adjust = phred_adjust

        self.fillna_val = 'N' if self.seq_type == 'NT' else 'X'

        self.encoding_setting = (encode_letters, 'utf-8')

        self._seq_to_table(seq_data)
        self._qual_to_table(quality_data, phred_adjust, return_table=False, nullshape=self.seq_arr.shape)
        self.contains_quality = self.qual_arr.shape[0] > 0


        # self.contains_quality = not(qualitydata is None)
        # self.seq_df = pd.DataFrame({'seqs': seqs, 'quals': qualitydata}) if qualitydata else pd.DataFrame({'seqs': seqs, 'quals': ''})

        # if isinstance(pos_names, int) or isinstance(pos_names, float):
        #     pos_names = np.arange(int(pos_names), int(pos_names) + st.shape[1])
        # elif pos_names is None:
        #     pos_names = np.arange(1, 1 + st.shape[1])
        # else:
        #     assert len(pos_names) == st.shape[1], 'ERROR the provided sequences and the provided position names are not the same size/shape!'
        #     pos_names = [p for p in pos_names]

        # d3array = (np.dstack((st.T, qualtable.T)).T).astype(np.uint8)
        # super(SeqPanel, self).__init__(d3array, items=['seqs', 'quals'], minor_axis=pos_names, major_axis=index)

        # self.seq_table = self.loc['seqs']
        # self.qual_table = self.loc['quals']



        # self.null_qual = null_qual
        # self.start = start
        # if seqtype not in ['AA', 'NT']:
        #     raise Exception('You defined seqtype as, {0}. We only allow seqtype to be "AA" or "NT"'.format(seqtype))
        # self.seqtype = seqtype
        # self.phred_adjust = phred_adjust
        # self.fillna_val = 'N' if seqtype == 'NT' else 'X'
        # self.loc = seqtable_indexer(self, 'loc')
        # self.iloc = seqtable_indexer(self, 'iloc')
        # self.ix = seqtable_indexer(self, 'ix')
        # self.encoding_setting = encode_letters  # (encode_letters, encoding)
        # if seqdata is not None:
        #     self.index = index
        #     self._seq_to_table(seqdata)
        #     self.qual_table = None
        #     if (isinstance(qualitydata, pd.Series) and qualitydata.empty is False) or qualitydata is not None:
        #         self.qual_to_table(qualitydata, phred_adjust, return_table=False)

    def __len__(self):
        return self.seq_list.shape[0]

    def slice_object(self, method, params):
        if method == 'loc':
            seq_table = self.seq_table.loc[params]
            qual_table = self.qual_table.loc[params] if self.qual_table is not None else None
            seq_df = self.seq_df.loc[params]
        elif method == 'iloc':
            seq_table = self.seq_table.iloc[params]
            qual_table = self.qual_table.iloc[params] if self.qual_table is not None else None
            seq_df = self.seq_df.iloc[params]
        elif method == 'ix':
            seq_table = self.seq_table.ix[params]
            qual_table = self.qual_table.ix[params] if self.qual_table is not None else None
            seq_df = self.seq_df.ix[params]
        if isinstance(seq_table, pd.Series):
            seq_table = pd.DataFrame(seq_table)
        if isinstance(qual_table, pd.Series):
            qual_table = pd.DataFrame(qual_table)
        if isinstance(seq_table, pd.Series):
            seq_table = pd.DataFrame(seq_table)
        try:
            return self.copy_using_template(seq_table, seq_df, qual_table)
        except:
            if isinstance(qual_table, pd.DataFrame):
                return self.copy_using_template(seq_table.transpose(), seq_df.transpose(), qual_table.transpose())
            else:
                return self.copy_using_template(seq_table.transpose(), seq_df.transpose(), None)

    def __getitem__(self, key):
        seq_table = self.seq_table.__getitem__(key)
        return self.copy_using_template(seq_table)

    def copy_using_template(self, template, template_seqdf=None, template_qual=None):
        new_member = seqtable(seq_type=self.seq_type, phred_adjust=self.phred_adjust, null_qual=self.null_qual)
        if template_qual is None and self.qual_table is not None:
            qual_table = self.qual_table.loc[template.index, template.columns]
        else:
            qual_table = template_qual
        if template_seqdf is None:
            self.slice_sequences(template.columns)
            seqs = self.slice_sequences(template.columns).loc[template.index]
        else:
            seqs = template_seqdf
        new_member.seq_df = seqs
        new_member.seq_table = template
        new_member.qual_table = qual_table
        new_member.index = template.index
        return new_member

    def view_bases(self, as_dataframe=False, side_by_side=False, num_base_show=10):
        np.set_printoptions(edgeitems=num_base_show)
        return self.seq_table.values.view('S1').T if side_by_side is True else self.seq_table.values.view('S1')

    def shape(self):
        return self.seq_table.shape

    def __repr__(self):
        return self.seq_df.__repr__()

    def __str__(self):
        return self.seq_table.__str__()

    def copy(self):
        return copy.deepcopy(self)

    def subsample(self, numseqs):
        """
            Return a random sample of sequences as a new object

            Args:
                numseqs (int): How many sequences to sample

            Returns:
                SeqTable Object
        """
        random_sequences = self.seq_df.sample(numseqs)

        if 'quals' in random_sequences:
            random_qualities = random_sequences['quals']
        else:
            random_qualities = None
        return seqtable(random_sequences['seqs'], random_qualities, self.start, index=random_sequences.index, seq_type=self.seq_type, phred_adjust=self.phred_adjust)

    def get_substrings(self, word_length, subsample_seqs=None, weights=None):
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
                >>> st = seq_tables.seqtable(['ACTW', 'ATTA'])
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

        tmp_table = self.seq_table if subsample_seqs is None else self.subsample(subsample_seqs).seq_table
        mapper = {c: i for i, c in enumerate(tmp_table.columns)}
        rev_mapper = {i: c for i, c in enumerate(tmp_table.columns)}
        substrings = [[mapper[x] for x in i] for i in itertools.combinations(list(tmp_table.columns), word_length)]
        table_values = tmp_table.values.view('S1')
        view_value = 'S' + str(word_length)

        dataframes = []

        if weights is None:
            for s in substrings:
                [a, b] = np.unique(table_values[:, s].reshape(-1).view(view_value), return_counts=True)
                dataframes.append(pd.DataFrame(b, index=a, columns=[tuple([rev_mapper[c] for c in s])]))
            substring_counts_df =  pd.concat(dataframes, axis=1).fillna(0)
        else:
            # for s in substrings:
            #     arr_vals = np.concatenate([table_values[:, s].reshape(-1).view(view_value).reshape(-1, 1), weights.reshape(-1, 1)], axis=1)
            #     arr_counts = dict_count(arr_vals)
            #     dataframes.append(pd.DataFrame(arr_counts.values(), index=arr_counts.keys(), columns=[tuple([rev_mapper[c] for c in s])]))
            for s in substrings:
                arr = table_values[:, s].reshape(-1).view(view_value)
                [a, b] = np.unique(arr, return_inverse=True)
                c = numpy_value_counts_bin_count(b, weights=weights).reset_index().values
                a = a[c[:, 0].astype(int)]
                b = c[:, 1]
                dataframes.append(pd.DataFrame(b, index=a, columns=[tuple([rev_mapper[c] for c in s])]))

            substring_counts_df = pd.concat(dataframes, axis=1).fillna(0)
            # dfs = pd.concat([
            #     pd.DataFrame(
            #         {
            #             'ss': table_values[:, s].reshape(-1).view(view_value),
            #             tuple([rev_mapper[c] for c in s]): weights
            #         }
            #     )
            #     for s in substrings
            # ], axis=0).fillna(0)
            # return dfs.groupby(by='ss').sum()
        if self.encoding_setting[0] is False:
            substring_counts_df.index = substring_counts_df.index.map(lambda x: x.decode())
        return substring_counts_df

    def update_seqdf(self):
        """
            Make seq_df attribute in sync with seq_table and qual_table
            Sometimes it might be useful to make changes to the seq_table attribute. For example, may you have your own custom code where you change the values of seq_table
            to be '.' or something random. Well you want to make sure that seq_df updates accordingly because the full length strings are the most useful in the end
        """
        self.seq_df = self.slice_sequences(self.seq_table.columns)

    def _qual_to_table(self, quals, phred_adjust=33, return_table=False, nullshape=(0, 0)):
        """
            Given a set of quality score strings, updates the  return a new dataframe such that each column represents the quality at each position as a number

            Args:
                quals: (Series or list of quality scores, default=None): If defined, then user is passing in quality data along with the sequences)
                phred_adjust (integer, default=33): If quality data is passed, then this will be used to adjust the quality score (i.e. Sanger vs older NGS quality scorning)
                return_table (boolean, default=False): If True, then the attribute self.qual_table is returned

            Returns:
                self.qual_table (Dataframe): each row corresponds to a specific sequence and each column corresponds to

        """
        if quals is None:
            # no quality scores
            self.qual_arr = np.array([]).reshape(*nullshape)
            return

        if hasattr(quals, 'values'):
            # its probably a dataframe/series
            # gotcha assumptions??? probably...
            quals = np.array(quals.values, dtype='S')
        else:
            quals = np.array(quals, dtype='S')

        self.qual_arr = strseries_to_bytearray(
            quals, self.null_qual,
            self.encoding_setting
        )

        assert self.qual_arr.shape == self.seq_arr.shape, \
            'Error the shape of the quality scores does not match the shape of the sequence scores: seq shape: {0}, qual shape {1}'.format(
                self.seq_arr.shape,
                self.qual_arr.shape
            )

    def _seq_to_table(self, seqlist):
        """
            Given a set of sequences, generates a dataframe such that each column represents a base or residue at each position of the aligned sequences

            .. important::Private function

                This function is not for public use
        """
        if hasattr(seqlist, 'values'):
            # its probably a dataframe/series
            # gotcha assumptions??? probably...
            seqlist = np.array(seqlist.values, dtype='S')
        else:
            seq_list = np.array(seqlist, dtype='S')

        self.seq_arr = strseries_to_bytearray(
            seq_list, self.fillna_val,
            self.encoding_setting
        )

    def table_to_seq(self, new_name):
        """
            Return the sequence list
        """
        raise 'NEED TO FIX!!!'

    def compare_to_reference(
            self, reference_seq, positions=None, ref_start=0, flip=False,
            set_diff=False, ignore_characters=[], treat_as_true=[], return_num_bases=False
    ):
        """
            Calculate which positions within a reference are not equal in all sequences in dataframe

            Args:
                reference_seq (string): A string that you want to align sequences to
                positions (list, default=None): specific positions in both the reference_seq and sequences you want to compare
                ref_start (int, default=0): where does the reference sequence start with respect to the aligned sequences
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

        # convert reference to numbers
        # reference_array = np.array(bytearray(reference_seq))[ref_cols]
        reference_array, compare_column_header = self.adjust_ref_seq(reference_seq, self.seq_table.columns, ref_start, positions, return_as_np=True)

        if set_diff is True:
            # change positions of interest to be the SET DIFFERENCE of positions parameter
            if positions is None:
                raise Exception('You cannot analyze the set-difference of all positions. Returns a non-informative answer (no columns to compare)')
            positions = sorted(list(set(compare_column_header) - set(positions)))
        else:
            # determine which columns we should look at
            if positions is None:
                ref_cols = [i for i in range(len(compare_column_header))]
                positions = compare_column_header
            else:
                positions = sorted(list(set(positions) & set(compare_column_header)))
                ref_cols = [i for i, c in enumerate(compare_column_header) if c in positions]

        # actually compare distances in each letter (find positions which are equal)
        diffs = self.seq_table[positions].values == reference_array[ref_cols]  # if flip is False else self.seq_table[positions].values != reference_array

        if treat_as_true:
            if not isinstance(treat_as_true, list):
                treat_as_true = [treat_as_true]
            treat_as_true = [ord(let) for let in treat_as_true]
            # now we have to ignore characters that are equal to specific values
            ignore_pos = (self.seq_table[positions].values == treat_as_true[0]) | (reference_array[ref_cols] == treat_as_true[0])
            for chr_p in range(1, len(treat_as_true)):
                ignore_pos = ignore_pos | (self.seq_table[positions].values == treat_as_true[chr_p]) | (reference_array[ref_cols] == treat_as_true[chr_p])

            # now adjust boolean results to ignore any positions == treat_as_true
            diffs = (diffs | ignore_pos)  # if flip is False else (diffs | ignore_pos)

        if flip:
            diffs = ~diffs

        if ignore_characters:
            if not isinstance(ignore_characters, list):
                ignore_characters = [ignore_characters]
            ignore_characters = [ord(let) for let in ignore_characters]
            # now we have to ignore characters that are equal to specific values
            ignore_pos = (self.seq_table[positions].values == ignore_characters[0]) | (reference_array[ref_cols] == ignore_characters[0])
            for chr_p in range(1, len(ignore_characters)):
                ignore_pos = ignore_pos | (self.seq_table[positions].values == ignore_characters[chr_p]) | (reference_array[ref_cols] == ignore_characters[chr_p])

            # OK so we need to FORCE np.nan, we cant do that if the datatype is a bool, so unfortunately we need to change the dattype
            # to be float in this situation
            df = pd.DataFrame(diffs, index=self.seq_table.index, dtype=float, columns=positions)
            df.values[ignore_pos] = np.nan
        else:
            # we will not need to replace nan anywhere, so we can use the smaller format of boolean here
            df = pd.DataFrame(diffs, index=self.seq_table.index, dtype=bool, columns=positions)

        if return_num_bases:
            #if ignore_characters:
            #    num_bases = len(positions) - ignore_pos.sum(axis=1)
            #else:
            #    num_bases = len(positions)
            num_bases = np.apply_along_axis(arr=df.values, axis=1, func1d=lambda x: len(x[~np.isnan(x)]))
            return df, num_bases
        else:
            return df

    def hamming_distance(self, reference_seq, positions=None, ref_start=0, set_diff=False, ignore_characters=[], normalized=False):
        """
            Determine hamming distance of all sequences in dataframe to a reference sequence.

            Args:
                reference_seq (string): A string that you want to align sequences to
                positions (list, default=None): specific positions in both the reference_seq and sequences you want to compare
                ref_start (int, default=0): where does the reference sequence start with respect to the aligned sequences
                set_diff (bool): If True, then we want to analyze positions that ARE NOT listed in positions parameters
                normalized (bool): If True, then divides hamming distance by the number of relevant bases
        """
        if normalized is True:
            diffs, bases = self.compare_to_reference(reference_seq, positions, ref_start, flip=True, set_diff=set_diff, ignore_characters=ignore_characters, return_num_bases=True)
            return pd.Series(diffs.fillna(0).values.sum(axis=1).astype(float) / bases, index=diffs.index)
        else:
            diffs = self.compare_to_reference(reference_seq, positions, ref_start, flip=True, set_diff=set_diff, ignore_characters=ignore_characters)
            return pd.Series(diffs.fillna(0).values.sum(axis=1), index=diffs.index)  # columns=c1, index=ind1)

    def mutation_profile(self, reference_seq, positions=None, ref_start=0, set_diff=False, ignore_characters=[], treat_as_true=[], normalized=False):
        """
            Return the type of mutation rates observed between the reference sequence and sequences in table.

            Args:
                reference_seq (string): A string that you want to align sequences to
                positions (list, default=None): specific positions in both the reference_seq and sequences you want to compare
                ref_start (int, default=0): where does the reference sequence start with respect to the aligned sequences
                set_diff (bool): If True, then we want to analyze positions that ARE NOT listed in positions parameters
                normalized (bool): If True, then frequency of each mutation
                ignore_characters: (char or list of chars): When performing distance/finding mismatches, always IGNORE THESE CHARACTERS, DONT TREAT THEM AS A MATCH OR A MISMATCH

            Returns:
                profile (pd.Series): Returns the counts (or frequency) for each mutation observed (i.e. A->C or A->T)
        """
        # def reference sequence
        ref = pd.DataFrame(
            self.adjust_ref_seq(reference_seq, self.seq_table.columns, ref_start, return_as_np=True, positions=positions)[0],
            index=self.seq_table.columns
        ).rename(columns={0: 'Ref base'}).transpose()
        # compare all bases/residues to the reference seq (returns a dataframe of boolean vars)
        not_equal_to = self.compare_to_reference(reference_seq, positions, ref_start, flip=True, treat_as_true=treat_as_true, set_diff=set_diff)
        # now create a numpy array in which the reference is repeated N times where n = # sequences
        ref = ref[not_equal_to.columns]
        ref_matrix = np.tile(ref, (self.seq_table.shape[0], 1))
        # now create a numpy array of ALL bases in the seq table that were not equal to the reference
        subset = self.seq_table[not_equal_to.columns]
        var_bases_unique = subset.values[(not_equal_to.values)]

        # now create a corresponding numpy array of ALL bases in teh REF TABLE where that base was not equal in the seq table
        # each index in this variable corresponds to the index (seq #, base position) in var_bases_unique
        ref_bases_unique = ref_matrix[(not_equal_to.values)]

        # OK lets do some fancy numpy methods and merge the two arrays, and then convert the 2D into 1D using bit conversion
        # found this at: https://www.reddit.com/r/learnpython/comments/3v9y8u/how_can_i_find_unique_elements_along_one_axis_of/
        mutation_combos = np.array([ref_bases_unique, var_bases_unique]).T.copy().view(np.int16)

        # finally count the instances of each mutation we see (use squeeze(1) to ONLY squeeze single dim)
        counts = np.bincount(mutation_combos.squeeze(1))
        unique_mut = np.nonzero(counts)[0]

        counts = counts[unique_mut]
        # convert values back to chacters of format (REF BASE/RESIDUE, VAR base/residue)
        unique_mut = unique_mut.astype(np.uint16).view(np.uint8).reshape(-1, 2).view('S1').astype('U1')

        # unique_mut, counts = np.unique(mutation_combos.squeeze(), return_counts=True) => this could have worked also, little slower
        if len(unique_mut) == 0:
            return pd.Series()
        mut_index = pd.MultiIndex.from_tuples(list(unique_mut), names=['ref', 'mut'])
        mutation_counts = pd.Series(index=mut_index, data=counts).astype(float).sort_index()

        del ref_bases_unique
        del var_bases_unique
        del mutation_combos

        if ignore_characters:
            # drop any of these mutation types => maybe i dont need to remove from axis of 0 (the provided reference)??
            mutation_counts = mutation_counts.unstack().drop(ignore_characters, axis=1, errors='ignore').drop(ignore_characters, axis=0, errors='ignore').stack()

        if normalized is True:
            mutation_counts = mutation_counts / (mutation_counts.sum())

        return mutation_counts

    def mutation_TS_TV_profile(self, reference_seq, positions=None, ref_start=0, set_diff=False, ignore_characters=[]):
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

        mutations = self.mutation_profile(reference_seq, positions, ref_start, set_diff, ignore_characters)
        if mutations.empty:
            return np.nan, 0, 0
        ts_freq = sum([mutations.loc[ts] for ts in transitions if ts in mutations.index]) / mutations.sum()
        tv_freq = sum([mutations.loc[tv] for tv in transversions if tv in mutations.index]) / mutations.sum()

        return ts_freq / tv_freq, ts_freq, tv_freq

    def mutation_profile_deprecated(self, reference_seq, positions=None, ref_start=0, set_diff=False, ignore_characters=[], normalized=False):
        """
            Return the type of mutation rates observed between the reference sequence and sequences in table.

            Args:
                reference_seq (string): A string that you want to align sequences to
                positions (list, default=None): specific positions in both the reference_seq and sequences you want to compare
                ref_start (int, default=0): where does the reference sequence start with respect to the aligned sequences
                set_diff (bool): If True, then we want to analyze positions that ARE NOT listed in positions parameters
                normalized (bool): If True, then frequency of each mutation

            Returns:
                profile (pd.Series): Returns the counts (or frequency) for each mutation observed (i.e. A->C or A->T)
                transversions (float): Returns frequency of transversion mutation
                transition (float): Returns frequency of transition mutation

                .. note::

                        Transversion and transitions only apply to situations when the seq_type is a NT
            .. note::

                This function has been deprecated because we found a better speed-optimized method
        """
        # def reference sequence
        ref = pd.DataFrame(self.adjust_ref_seq(reference_seq, self.seq_table.columns, ref_start, return_as_np=True)[0], index=self.seq_table.columns, positions=positions).rename(columns={0: 'Ref base'})
        # compare all bases/residues to the reference seq (returns a dataframe of boolean vars)
        not_equal_to = self.compare_to_reference(reference_seq, positions, ref_start, flip=True, set_diff=set_diff, ignore_characters=ignore_characters)
        subset = self.seq_table[not_equal_to.columns]
        # stack all mutations that are NOT equal to the reference
        # this creates a dataframe such that each row is essentially seqpos, base position: letter at that position
        # delete the level_0 (seqpos) because its trivial to analysis
        mutation_counts = pd.DataFrame(subset[not_equal_to].stack()).reset_index().rename(columns={0: 'Var base', 'level_1': 'Pos'}).astype(int).drop('level_0', axis=1)
        # now merge the results from the reference bases, once merge, we can count unique occurrences of ref base -> var base
        mutation_counts = mutation_counts.merge(ref, left_on='Pos', right_index=True, how='inner')
        # convert columns  to letters rather than ascii
        mutation_counts = mutation_counts.groupby(by=['Ref base', 'Var base']).apply(len).reset_index()
        mutation_counts[['Ref base', 'Var base']] = mutation_counts[['Ref base', 'Var base']].applymap(lambda x: chr(x))
        mutation_counts = mutation_counts.set_index(['Ref base', 'Var base'])[0]
        if normalized is True:
            mutation_counts = mutation_counts / (1.0 * mutation_counts.sum())
        return mutation_counts

    def quality_filter(self, q, p, inplace=False, ignore_null_qual=True):
        """
            Filter out sequences based on their average qualities at each base/position

            Args:
                q (int): quality score cutoff
                p (int/float/percent 0-100): the percent of bases that must have a quality >= the cutoff q
                inplace (boolean): If False, returns a copy of the object filtered by quality score
                ignore_null_qual (boolean): Ignore bases that are not represented. (i.e. those with quality of 0)
        """
        if self.qual_table is None:
            raise Exception("You have not passed in any quality data for these sequences")

        meself = self if inplace is True else copy.deepcopy(self)
        total_bases = (meself.qual_table.values > (ord(self.null_qual) - self.phred_adjust)).sum(axis=1) if ignore_null_qual else meself.qual_table.shape[1]
        percent_above = (100 * ((meself.qual_table.values >= q).sum(axis=1))) / total_bases

        meself.qual_table = meself.qual_table[percent_above >= p]
        meself.seq_table = meself.seq_table.loc[meself.qual_table.index]
        meself.seq_df = meself.seq_df.loc[meself.qual_table.index]
        # bases = meself.seq_table.shape[1]

        if inplace is False:
            return meself

    def convert_low_bases_to_null(self, q, replace_with='N', inplace=False):
        """
            This will convert all letters whose corresponding quality is below a cutoff to the value replace_with

            Args:
                q (int): quality score cutoff, convert all bases whose quality is < than q
                inplace (boolean): If False, returns a copy of the object filtered by quality score
                replace_with (char): a character to replace low bases with
        """
        if self.qual_table is None:
            raise Exception("You have not passed in any quality data for these sequences")

        meself = self if inplace is True else self.copy()
        replace_with = ord(replace_with) if replace_with is not None else ord('N') if self.seq_type == 'NT' else ord('X')
        meself.seq_table.values[meself.qual_table.values < q] = replace_with
        chars = self.seq_table.shape[1]
        meself.seq_df['seqs'] = list(meself.seq_table.values.copy().view('S' + str(chars)).ravel())
        if inplace is False:
            return meself

    def adjust_ref_seq(self, ref, table_columns, ref_start, positions, return_as_np=True):
            """
            Aligns a reference sequence such that its position matches positions within the seqtable of interest

            Args:
                ref (str): Represents the reference sequence
                table_columns (list or series): Defines the column positions or column names
                ref_start (int): Defines where the reference starts relative to the sequence

            """
            compare_column_header = list(table_columns)
            reference_seq = ref.upper()
            if ref_start < 0:
                # simple: the reference sequence is too long, so just trim it
                reference_seq = reference_seq[(-1 * ref_start):]
            elif ref_start > 0:
                reference_seq = self.fillna_val * ref_start + reference_seq
                # more complicated because we need to return results to user in the way they expected. What to do if the poisitions they requested are not
                # found in reference sequence provided
                if positions is None:
                    positions = compare_column_header
                # ignore_postions = compare_column_header[ref_start]
                before_filter = positions
                positions = [p for p in positions if p >= ref_start]
                if len(positions) < len(before_filter):
                    warnings.warn("Warning: Because the reference starts at a position after the start of sequences we cannot anlayze the following positions: {0}".format(','.join([_ for _ in before_filter[:ref_start]])))
                compare_column_header = compare_column_header[ref_start:]

            if len(reference_seq) > len(table_columns):
                reference_seq = reference_seq[:len(table_columns)]
            elif len(reference_seq) < len(table_columns):
                reference_seq = reference_seq + self.fillna_val * (self.seq_table.shape[1] - len(reference_seq))

            return np.array([reference_seq], dtype='S').view(np.uint8) if return_as_np is True else reference_seq, compare_column_header

    def slice_sequences(self, positions, name='seqs', return_quality=False, empty_chars=None):
        if empty_chars is None:
            empty_chars = self.fillna_val

        positions = [p for p in positions]
        num_chars = len(positions)

        # confirm that all positions are present in the column
        missing_pos = set(positions) - set(self.seq_table.columns)

        if len(missing_pos) > 0:
            new_positions = [p for p in positions if p in self.seq_table.columns]
            prepend = ''.join([empty_chars for p in positions if p < self.seq_table.columns[0]])
            append = ''.join([empty_chars for p in positions if p > self.seq_table.columns[-1]])
            positions = new_positions
            num_chars = len(positions)
            warnings.warn("The sequences do not cover all positions requested. {0}'s will be appended and prepended to sequences as necessary".format(empty_chars))
        else:
            prepend = ''
            append = ''

        if positions == []:
            if return_quality:
                qual_empty = '!' * (len(prepend) + len(append))
                return pd.DataFrame({'seqs': prepend + append, 'quals': qual_empty}, columns=['seqs', 'quals'], index=self.index)
            else:
                return pd.DataFrame(prepend + append, columns=['seqs'], index=self.index)

        substring = pd.DataFrame({name: self.seq_table.loc[:, positions].values.copy().view('S{0}'.format(num_chars)).ravel()}, index=self.index)

        if prepend or append:
            substring['seqs'] = substring['seqs'].apply(lambda x: prepend + x + append)

        if self.qual_table is not None and return_quality:
            subquality = self.qual_table.loc[:, positions].values
            subquality = (subquality + self.phred_adjust).copy().view('S{0}'.format(num_chars)).ravel()
            substring['quals'] = subquality
            if prepend or append:
                prepend = '!' * len(prepend)
                append = '!' * len(append)
                substring['quals'] = substring['quals'].apply(lambda x: prepend + x + append)

        return substring

    def get_seq_dist(self, positions=None, method='counts', ignore_characters=[], weight_by=None, ):
        """
            Returns the distribution of bases or amino acids at each position.
        """
        if weight_by is not None:
            try:
                if isinstance(weight_by, pd.Series):
                    assert(weight_by.shape[0] == self.seq_table.shape[0])
                    weight_by = weight_by.values
                elif isinstance(weight_by, pd.DataFrame):
                    assert(weight_by.shape[0] == self.seq_table.shape[0])
                    assert(weight_by.shape[1] == 1)
                    weight_by = weight_by.values
                else:
                    assert(len(weight_by) == self.seq_table.shape[0])
                    weight_by = np.array(weight_by)
            except:
                raise Exception('The provided weights for each seuence must match the number of input sequences!')
        compare = self.seq_table.loc[:, positions] if positions else self.seq_table

        column_names = compare.columns
        dist = numpy_value_counts_bin_count(compare, weight_by)   # compare.apply(pd.value_counts).fillna(0)
        dist.rename({c: chr(c) for c in list(dist.index)}, inplace=True)
        drop_values = list(set(ignore_characters) & set(list(dist.index)))
        dist = dist.drop(drop_values, axis=0)
        if method == 'freq':
            dist = dist.astype(float) / dist.sum(axis=0)
        elif method == 'bits':
            N = self.seq_table.shape[0]
            dist = get_bits(dist.astype(float) / dist.sum(axis=0), self.seq_type, N)
        dist.rename(columns={old: new for (old, new) in zip(dist.columns, column_names)}, inplace=True)
        return dist.fillna(0)

    def get_plogo(self, background_seqs=None, positions=None, ignore_characters=[], alpha=0.01):
        counts = self.get_seq_dist(positions, ignore_characters=ignore_characters)
        if background_seqs is not None:
            bkst = seqtable(background_seqs, seq_type=self.seq_type)
            bkst_freq = bkst.get_seq_dist(positions, method='freq', ignore_characters=ignore_characters)
        else:
            bkst_freq = None

        return get_plogo(counts, self.seq_type, bkst_freq, alpha=alpha)

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

    def pos_entropy(self, positions=None, ignore_characters=[], nbit=2):
        dist = self.get_seq_dist(positions, method='freq', ignore_characters=ignore_characters)
        return shannon_info(dist, nbit)

    def relative_entropy(self, background_seqs=None, positions=None, ignore_characters=[]):
        dist = self.get_seq_dist(positions, method='freq', ignore_characters=ignore_characters)
        if background_seqs is not None:
            bkst = seqtable(background_seqs, seq_type=self.seq_type)
            bkst_freq = bkst.get_seq_dist(positions, 'freq', ignore_characters)
        else:
            bkst_freq = None
        return relative_entropy(dist, self.seq_type, bkst_freq)

    def seq_logo(self, positions=None, weights=None, method='freq', ignore_characters=[], **kwargs):
        dist = self.get_seq_dist(positions, method, ignore_characters, weights)
        return draw_seqlogo_barplots(dist, alphabet=self.seq_type, **kwargs)

    def get_quality_dist(self, bins='fastqc', percentiles=[10, 25, 50, 75, 90], exclude_null_quality=True, sample=None, plotly_sampledata_size=20):
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
        assert (self.qual_table is not None)
        return get_quality_dist(self.qual_table, bins, percentiles, exclude_null_quality, sample, plotly_sampledata_size=20)


class seqtable_indexer():
    def __init__(self, obj, method):
        self.obj = obj
        self.method = method

    def __getitem__(self, key):
        return self.obj.slice_object(self.method, key)
