"""
**We can use Pandas to analyze aligned sequences in a table. This can be useful for quickly generating AA or NT distribution by position
and accessing specific positions of an aligned sequence**
"""
import gc
import copy
import warnings
import pandas as pd
import math
import numpy as np

try:
    from Bio import SeqIO
    bio_installed = True
except:
    bio_installed = False

# from collections import defaultdict
from seq_logo import draw_seqlogo_barplots
from seq_table_util import get_quality_dist  # , degen_to_base, dna_alphabet, aa_alphabet


def strseries_to_bytearray(series, fillvalue):
    max_len = series.apply(len).max()
    series = series.apply(lambda x: x.ljust(max_len, fillvalue))
    seq_as_int = np.array(list(series), dtype='S').view('S1').reshape((series.size, -1)).view('uint8')
    return (series, seq_as_int)


class seqtable():
    """
    Class for viewing aligned sequences within a list or dataframe. This will take a list of sequences and create views such that
    we can access each position within specific positions. It will also associate quality scores for each base if provided.

    Args:
        seqdata (Series, or list of strings): List containing a set of sequences aligned to one another
        qualitydata (Series or list of quality scores, default=None): If defined, then user is passing in quality data along with the sequences)
        start (int): Explicitly define where the aligned sequences start with respect to some refernce frame (i.e. start > 2 means sequences start at position 2 not 1)
        index (list of values defining the index, default=None):

            .. note::Index=None

                If None, then the index will result in default integer indexing by pandas.

        seqtype (string of 'AA' or 'NT', default='NT'): Defines the format of the data being passed into the dataframe
        phred_adjust (integer, default=33): If quality data is passed, then this will be used to adjust the quality score (i.e. Sanger vs older NGS quality scorning)

    Attributes:
        seq_df (Dataframe): Each row in the dataframe is a sequence. It will always contain a 'seqs' column representing the sequences past in. Optionally it will also contain a 'quals' column representing quality scores
        seq_table (Dataframe): Dataframe representing sequences as characters in a table. Each row in the dataframe is a sequence. Each column represents the position of a base/residue within the sequence. The 4th position of sequence 2 is found as seq_table.ix[1, 4]
        qual_table (Dataframe, optional): Dataframe representing the quality score for each character in seq_table

    Examples:
        >>> sq = seq_tables.seqtable(['AAA', 'ACT', 'ACA'])
        >>> sq.hamming_distance('AAA')
        >>> sq = read_fastq('fastqfile.fq')
    """
    def __init__(self, seqdata=None, qualitydata=None, start=1, index=None, seqtype='NT', phred_adjust=33, null_qual='!', **kwargs):
        self.null_qual = null_qual
        self.start = start
        if seqtype not in ['AA', 'NT']:
            raise Exception('You defined seqtype as, {0}. We only allow seqtype to be "AA" or "NT"'.format(seqtype))
        self.seqtype = seqtype
        self.phred_adjust = phred_adjust
        self.fillna_val = 'N' if seqtype == 'NT' else 'X'
        self.loc = seqtable_indexer(self, 'loc')
        self.iloc = seqtable_indexer(self, 'iloc')
        self.ix = seqtable_indexer(self, 'ix')
        if seqdata is not None:
            self.index = index
            self._seq_to_table(seqdata)
            self.qual_table = None
            if (isinstance(qualitydata, pd.Series) and qualitydata.empty is False) or qualitydata is not None:
                self.qual_to_table(qualitydata, phred_adjust, return_table=False)

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
        new_member = seqtable(seqtype=self.seqtype, phred_adjust=self.phred_adjust, null_qual=self.null_qual)
        if template_qual is None and self.qual_table is not None:
            qual_table = self.qual_table.loc[template.index, template.columns]
        else:
            qual_table = None
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
        return seqtable(random_sequences['seqs'], random_qualities, self.start, index=random_sequences.index, seqtype=self.seqtype, phred_adjust=self.phred_adjust)

    def update_seqdf(self):
        """
            Make seq_df attribute in sync with seq_table and qual_table
            Sometimes it might be useful to make changes to the seq_table attribute. For example, may you have your own custom code where you change the values of seq_table
            to be '.' or something random. Well you want to make sure that seq_df updates accordingly because the full length strings are the most useful in the end
        """
        self.seq_df = self.slice_sequences(self.seq_table.columns)

    def qual_to_table(self, qualphred, phred_adjust=33, return_table=False):
        """
            Given a set of quality score strings, updates the  return a new dataframe such that each column represents the quality at each position as a number

            Args:
                qualphred: (Series or list of quality scores, default=None): If defined, then user is passing in quality data along with the sequences)
                phred_adjust (integer, default=33): If quality data is passed, then this will be used to adjust the quality score (i.e. Sanger vs older NGS quality scorning)
                return_table (boolean, default=False): If True, then the attribute self.qual_table is returned

            Returns:
                self.qual_table (Dataframe): each row corresponds to a specific sequence and each column corresponds to

        """
        if isinstance(qualphred, list):
            qual_list = pd.Series(qualphred)
        else:
            qual_list = qualphred
        (qual_list, self.qual_table) = strseries_to_bytearray(qual_list, self.null_qual)

        self.seq_df['quals'] = list(qual_list)
        self.qual_table -= self.phred_adjust

        self.qual_table = pd.DataFrame(self.qual_table, index=self.index, columns=range(self.start, self.qual_table.shape[1] + self.start))

        if self.qual_table.shape != self.seq_table.shape:
            raise Exception("The provided quality list does not match the format of the sequence list. Shape of sequences {0}, shape of quality {1}".format(str(self.seq_table.shape), str(self.qual_table.shape)))
        return self.qual_table

    def _seq_to_table(self, seqlist):
        """
            Given a set of sequences, generates a dataframe such that each column represents a base or residue at each position of the aligned sequences

            .. important::Private function

                This function is not for public use
        """
        if isinstance(seqlist, list):
            seq_list = pd.Series(seqlist)
        else:
            seq_list = seqlist

        (seq_list, self.seq_table) = strseries_to_bytearray(seq_list, self.fillna_val)
        self.seq_df = pd.DataFrame(list(seq_list), index=self.index, columns=['seqs'])

        self.seq_table = pd.DataFrame(self.seq_table, index=self.index, columns=range(self.start, self.seq_table.shape[1] + self.start))

    def table_to_seq(self, new_name):
        """
            Return the sequence list
        """
        return self.seq_list

    def compare_to_reference(self, reference_seq, positions=None, ref_start=0, flip=False, set_diff=False, ignore_characters=[], return_num_bases=False):
        """
            Calculate which positions within a reference are not equal in all sequences in dataframe

            Args:
                reference_seq (string): A string that you want to align sequences to
                positions (list, default=None): specific positions in both the reference_seq and sequences you want to compare
                ref_start (int, default=0): where does the reference sequence start with respect to the aligned sequences
                flip (bool): If True, then find bases that ARE MISMATCHES(NOT equal) to the reference
                set_diff (bool): If True, then we want to analyze positions that ARE NOT listed in positions parameters
                ignore_characters (char or list of chars): When performing distance/finding mismatches, always treat these characters as matches
                return_num_bases (bool): If true, returns a second argument defining the number of relevant bases present in each row

                    ..important:: Change in output results

                        Setting return_num_bases to true will change how results are returned (two elements rather than one are returned)

            Returns:
                Dataframe of boolean variables showing whether base is equal to reference at each position
        """

        # reference_seq = reference_seq.upper()
        # compare_column_header = list(self.seq_table.columns)
        # if ref_start < 0:
        #     # simple: the reference sequence is too long, so just trim it
        #     reference_seq = reference_seq[(-1 * ref_start):]
        # elif ref_start > 0:
        #     reference_seq = self.fillna_val * ref_start + reference_seq
        #     # more complicated because we need to return results to user in the way they expected. What to do if the poisitions they requested are not
        #     # found in reference sequence provided
        #     if positions is None:
        #         positions = compare_column_header
        #     ignore_postions = compare_column_header[ref_start]
        #     before_filter = positions
        #     positions = [p for p in positions if p >= ref_start]
        #     if len(positions) < len(before_filter):
        #         warnings.warn("Warning: Because the reference starts at a position after the start of sequences we cannot anlayze the following positions: {0}".format(','.join([_ for _ in before_filter[:ref_start]])))
        #     compare_column_header = compare_column_header[ref_start:]

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

        if ignore_characters:
            if not isinstance(ignore_characters, list):
                ignore_characters = [ignore_characters]
            ignore_characters = [ord(let) for let in ignore_characters]
            # now we have to ignore characters that are equal to specific values
            ignore_pos = (self.seq_table[positions].values == ignore_characters[0]) | (reference_array[ref_cols] == ignore_characters[0])
            for chr_p in range(1, len(ignore_characters)):
                ignore_pos = ignore_pos | (self.seq_table[positions].values == ignore_characters[chr_p]) | (reference_array[ref_cols] == ignore_characters[chr_p])

            # now adjust boolean results to ignore any positions == ignore_characters
            diffs = (diffs | ignore_pos)  # if flip is False else (diffs | ignore_pos)

        if flip:
            diffs = ~diffs

        if return_num_bases:
            if ignore_characters:
                num_bases = len(positions) - ignore_pos.sum(axis=1)
            else:
                num_bases = len(positions)
            return pd.DataFrame(diffs, index=self.seq_table.index, dtype=bool, columns=positions), num_bases
        else:
            return pd.DataFrame(diffs, index=self.seq_table.index, dtype=bool, columns=positions)

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
            return pd.Series(diffs.values.sum(axis=1).astype(float) / bases, index=diffs.index)
        else:
            diffs = self.compare_to_reference(reference_seq, positions, ref_start, flip=True, set_diff=set_diff, ignore_characters=ignore_characters)
            return pd.Series(diffs.values.sum(axis=1), index=diffs.index)  # columns=c1, index=ind1)

    def mutation_profile(self, reference_seq, positions=None, ref_start=0, set_diff=False, ignore_characters=[], normalized=False):
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
        """
        # def reference sequence
        ref = pd.DataFrame(self.adjust_ref_seq(reference_seq, self.seq_table.columns, ref_start, return_as_np=True, positions=positions)[0], index=self.seq_table.columns).rename(columns={0: 'Ref base'}).transpose()
        # compare all bases/residues to the reference seq (returns a dataframe of boolean vars)
        not_equal_to = self.compare_to_reference(reference_seq, positions, ref_start, flip=True, set_diff=set_diff, ignore_characters=ignore_characters)
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
        unique_mut = unique_mut.astype(np.uint16).view(np.uint8).reshape(-1, 2).view('S1')
        # unique_mut, counts = np.unique(mutation_combos.squeeze(), return_counts=True) => this could have worked also, little slower
        if len(unique_mut) == 0:
            return pd.Series()
        mut_index = pd.MultiIndex.from_tuples(list(unique_mut), names=['ref', 'mut'])
        mutation_counts = pd.Series(index=mut_index, data=counts).astype(float).sort_index()
        if normalized is True:
            mutation_counts = mutation_counts / (mutation_counts.sum())
        del ref_bases_unique
        del var_bases_unique
        del mutation_combos

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
        if self.seqtype != 'NT':
            raise('Error: you cannot calculate TS and TV mutations on AA sequences. Either the seqtype is incorrect or you want to use the function mutation_profile')
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

                        Transversion and transitions only apply to situations when the seqtype is a NT
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

        self.shape = self.seq_table.shape
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
        replace_with = ord(replace_with) if replace_with is not None else ord('N') if self.seqtype == 'NT' else ord('X')
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

    def get_seq_dist(self, positions=None, method='counts', ignore_characters=[],):
        """
            Returns the distribution of bases or amino acids at each position.
        """
        compare = self.seq_table.loc[:, positions] if positions else self.seq_table
        dist = compare.apply(pd.value_counts).fillna(0)
        dist.rename({c: chr(c) for c in list(dist.index)}, inplace=True)
        drop_values = list(set(ignore_characters) & set(list(dist.index)))
        dist = dist.drop(drop_values, axis=0)
        if method == 'freq':
            dist = dist.astype(float) / dist.sum(axis=0)

        return dist.fillna(0)

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

    def seq_logo(self, positions=None, method='freq', ignore_characters=[], **kwargs):
        dist = self.get_seq_dist(positions, method, ignore_characters)
        return draw_seqlogo_barplots(dist, alphabet=self.seqtype, **kwargs)

    def get_quality_dist(self, bins=None, percentiles=[0.1, 0.25, 0.5, 0.75, 0.9], exclude_null_quality=True, sample=None):
        """
            Returns the distribution of quality across the given sequence, similar to FASTQC quality seq report.

            Args:
                bins(list of ints or tuples, default=None): bins defines how to group together the columns/sequence positions when aggregating the statistics.

                    .. note:: bins=None

                        If bins is none, then by default, bins are set to the same ranges defined by fastqc report

                percentiles (list of floats, default=[0.1, 0.25, 0.5, 0.75, 0.9]): value passed into pandas describe() function.
                exclude_null_quality (boolean, default=True): do not include quality scores of 0 in the distribution
                sample (int, default=None): If defined, then we will only calculate the distribution on a random subsampled population of sequences

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
        return get_quality_dist(self.qual_table, bins, percentiles, exclude_null_quality, sample)


def read_fastq(input_file, limit=None, chunk_size=10000, use_header_as_index=True, use_pandas=True):
    """
        Load a fastq file as class SeqTable
    """
    def group_fastq(index):
        return index % 4
    if use_pandas:
        grouped = pd.read_csv(input_file, sep='\n', header=None).groupby(group_fastq)
        header = list(grouped.get_group(0)[0].apply(lambda x: x[1:]))
        seqs = list(grouped.get_group(1)[0])
        quals = list(grouped.get_group(3)[0])
    else:
        if bio_installed is False:
            raise Exception("You do not have BIOPYTHON installed and therefore must use pandas to read the fastq (set use_pandas parameter to True). If you would like to use biopython then please install")
        seqs = []
        quals = []
        header = []
        for seq in SeqIO.parse(input_file, 'fastq'):
            r = seq.format('fastq').split('\n')
            header.append(r[0])
            seqs.append(r[1])
            quals.append(r[3])
    return seqtable(seqs, quals, index=header, seqtype='NT')


def read_sam(input_file, limit=None, chunk_size=100000, cleave_softclip=False, use_header_as_index=True):
    """
        Load a SAM file into class SeqTable
    """
    skiplines = 0
    with open(input_file) as r:
        for i in r:
            if i[0] == '@':
                skiplines += 1
            else:
                break

    cols_to_use = [9, 10]
    if use_header_as_index:
        cols_to_use.append(0)
        index_col = 0
    else:
        index_col = None

    if cleave_softclip:
        cols_to_use.append(5)

    df = pd.read_csv(input_file, sep='\t', header=None, index_col=index_col, usecols=cols_to_use, skiprows=skiplines)
    index = df.index
    return seqtable(df[9], df[10], index, 'NT')


class seqtable_indexer():
    def __init__(self, obj, method):
        self.obj = obj
        self.method = method

    def __getitem__(self, key):
        return self.obj.slice_object(self.method, key)
