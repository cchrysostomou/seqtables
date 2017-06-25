import xarray as xr
import numpy as np
import warnings
import pandas as pd


@xr.register_dataset_accessor('st')
class SeqTablesAccessor(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        st_attrs = self._obj.attrs['seqtable'].copy() if 'seqtable' in self._obj.attrs else {}
        if 'references' not in st_attrs:
            st_attrs['references'] = {}
        self._phred_adjust = st_attrs['phred_adjust'] if 'phred_adjust' in st_attrs else 33
        self._fill_na = st_attrs['fillna_val'] if 'fillna_val' in st_attrs else 'N'
        self.seq_type = st_attrs['seq_type'] if 'seq_type' in st_attrs else 'NT'
        if 'default_ref' not in st_attrs:
            if len(st_attrs['references']) == 0:
                self.default_ref = None
            else:
                self.default_ref = list(st_attrs['references'].keys())[0]
        else:
            self.default_ref = st_attrs['default_ref']

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

    def compare_to_reference(
        self, reference_seq, use_ref=None, positions=None, ref_start=0, flip=False,
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

        if use_ref is None:
            use_ref = self.default_ref

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
            num_bases = np.apply_along_axis(arr=df.values, axis=1, func1d=lambda x: len(x[~np.isnan(x)]))
            return df, num_bases
        else:
            return df