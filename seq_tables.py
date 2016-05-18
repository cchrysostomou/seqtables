"""
**We can use Pandas to analyze aligned sequences in a table. This can be useful for quickly generating AA or NT distribution by position
and accessing specific positions of an aligned sequence**
"""

import pandas as pd
import numpy as np
from plotly import graph_objs as go
from Bio import SeqIO
import gc
import copy
import warnings

degen_to_base = {
    'GT': 'K',
    'AC': 'M',
    'ACG': 'V',
    'CGT': 'B',
    'AG': 'R',
    'AGT': 'D',
    'A': 'A',
    'CG': 'S',
    'AT': 'W',
    'T': 'T',
    'C': 'C',
    'G': 'G',
    'ACGT': 'N',
    'ACT': 'H',
    'CT': 'Y'
}

dna_alphabet = list('ACTG') + list(set(sorted(degen_to_base.values())) - set('ACTG')) + ['-.']
aa_alphabet = list('ACDEFGHIKLMNPQRSTVWYX*Z-.')


def strseries_to_bytearray(series, fillvalue):
    max_len = series.apply(len).max()
    series = series.apply(lambda x: x.ljust(max_len, fillvalue))
    seq_as_int = np.array(list(series)).view('S1').reshape((series.size, -1)).view(np.int8)
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
        if seqdata is not None:
            self.index = index
            self._seq_to_table(seqdata)
            self.qual_table = None
            if (isinstance(qualitydata, pd.Series) and qualitydata.empty is False) or qualitydata is not None:
                self.qual_to_table(qualitydata, phred_adjust, return_table=False)

    def __len__(self):
        return self.seq_list.shape[0]

    def __getitem__(self, key):
        seq_table = self.seq_table.__getitem__(key)
        return self.copy_using_template(seq_table)

    def loc(self, key):
        pass
        # stable = self.seq_table.loc(key)
        # if self.qual_table:
        #     qtable = self.qual_table.__getitem__(key)
        # else:
        #     qtable = None
        # return SeqTable(None, copy_constructor={'seqs': stable, 'quals': qtable, 'seqtype': self.seqtype, 'phred_adjust': self.phred_adjust, 'memory_safe': self.memory_safe, 'alphabet': self.alphabet})

    def copy_using_template(self, template):
        new_member = seqtable(seqtype=self.seqtype, phred_adjust=self.phred_adjust, null_qual=self.null_qual)
        if self.qual_table is not None:
            qual_table = self.qual_table.loc[template.index, template.columns]
        else:
            qual_table = None
        seqs = self.slice_sequences(template.columns).loc[template.index]
        new_member.seq_df = seqs
        new_member.seq_table = template
        new_member.qual_table = qual_table
        new_member.index = template.index
        return new_member

    def view_bases(self, as_dataframe=False):
        return self.seq_table.values.view('S1')

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

    def compare_to_reference(self, reference_seq, positions=None, ref_start=0, flip=False, set_diff=False, ignore_characters=[]):
        """
            Calculate which positions within a reference are not equal in all sequences in dataframe

            Args:
                reference_seq (string): A string that you want to align sequences to
                positions (list, default=None): specific positions in both the reference_seq and sequences you want to compare
                ref_start (int, default=0): where does the reference sequence start with respect to the aligned sequences
                flip (bool): If True, then find bases that ARE MISMATCHES(NOT equal) to the reference
                set_diff (bool): If True, then we want to analyze positions that ARE NOT listed in positions parameters
                ignore_characters (char or list of chars): When performing distance/finding mismatches, always treat these characters as matches

            Returns:
                Dataframe of boolean variables showing whether base is equal to reference at each position
        """

        compare_column_header = self.seq_table.columns
        if ref_start < 0:
            # simple: the reference sequence is too long, so just trim it
            reference_seq = reference_seq[(-1 * ref_start):]
        elif ref_start > 0:
            reference_seq = self.fillna_val * ref_start + reference_seq
            # more complicated because we need to return results to user in the way they expected. What to do if the poisitions they requested are not
            # found in reference sequence provided
            if positions is None:
                positions = compare_column_header
            ignore_postions = compare_column_header[ref_start]
            before_filter = positions
            positions = [p for p in positions if p >= ref_start]
            if len(positions) < len(before_filter):
                warnings.warn("Warning: Because the reference starts at a position after the start of sequences we cannot anlayze the following positions: {0}".format(','.join([_ for _ in before_filter[:ref_start]])))
            compare_column_header = compare_column_header[ref_start:]

        # adjust reference length
        if len(reference_seq) > self.seq_table.shape[1]:
            reference_seq = reference_seq[:self.seq_table.shape[1]]
        elif len(reference_seq) < self.seq_table.shape[1]:
            reference_seq = reference_seq + self.fillna_val * (self.seq_table.shape[1] - len(reference_seq))

        if set_diff is True:
            # change positions of interest to be the SET DIFFERENCE of positions parameter
            if positions is None:
                raise Exception('You cannot analyze the setdifferene of all positions. Returns a non-informative answer (no columns to compare)')
            positions = list(set(compare_column_header) - set(positions))

        # determine which columns we should look at
        if positions is None:
            ref_cols = [i for i in range(len(compare_column_header))]
            positions = compare_column_header
        else:
            ref_cols = [i for i, c in enumerate(compare_column_header) if c in positions]

        # convert reference to numbers
        reference_array = np.array(bytearray(reference_seq))[ref_cols]
        # actually compare distances in each letter (find positions which are equal)
        diffs = self.seq_table[positions].values == reference_array  # if flip is False else self.seq_table[positions].values != reference_array

        if ignore_characters:
            if not isinstance(ignore_characters, list):
                ignore_characters = [ignore_characters]
            ignore_characters = [ord(let) for let in ignore_characters]
            # now we have to ignore characters that are equal to specific values
            ignore_pos = (self.seq_table[positions].values == ignore_characters[0]) | (reference_array == ignore_characters[0])
            for chr_p in range(1, len(ignore_characters)):
                ignore_pos = ignore_pos | (self.seq_table[positions].values == ignore_characters[chr_p]) | (reference_array == ignore_characters[chr_p])

            # now adjust boolean results to ignore any positions == ignore_characters
            diffs = (diffs | ignore_pos)  # if flip is False else (diffs | ignore_pos)

        if flip:
            diffs = ~diffs

        return pd.DataFrame(diffs, index=self.seq_table.index, dtype=bool)

    def hamming_distance(self, reference_seq, positions=None, ref_start=0, set_diff=False, ignore_characters=[]):
        """
            Determine hamming distance of all sequences in dataframe to a reference sequence.

            Args:
                reference_seq (string): A string that you want to align sequences to
                positions (list, default=None): specific positions in both the reference_seq and sequences you want to compare
                ref_start (int, default=0): where does the reference sequence start with respect to the aligned sequences
                set_diff (bool): If True, then we want to analyze positions that ARE NOT listed in positions parameters
        """
        diffs = self.compare_to_reference(reference_seq, positions, ref_start, flip=True, set_diff=set_diff, ignore_characters=ignore_characters)
        return diffs.sum(axis=1)

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
        bases = meself.seq_table.shape[1]

        self.shape = self.seq_table.shape
        if inplace is False:
            return meself


    def convert_low_bases_to_null(self, q, replace_with=None, inplace=False):
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
        meself.seq_table[meself.qual_table < q] = replace_with
        chars = self.seq_table.shape[1]
        meself.seq_df['seqs'] = list(meself.seq_table.values.copy().view('S' + str(chars)).ravel())
        if inplace is False:
            return meself

    def slice_sequences(self, positions, name='seqs', return_quality=False):
        positions = [p for p in positions]
        num_chars = len(positions)
        
        if num_chars > len(self.seq_table.columns):
            new_positions = [p for p in positions if p in self.seq_table.columns]
            prepend = ''.join(['N' for p in positions if p < self.seq_table.columns[0]])
            append = ''.join(['N' for p in positions if p > self.seq_table.columns[-1]])
            positions = new_positions
            num_chars = len(positions)
            warnings.warn("The sequences do not cover all positions requested. N's will be appended and prepended to sequences as necessary")
        else:
            prepend = ''
            append = ''

        if positions == []:
            if return_quality:
                qual_empty = '!' * (len(prepend) + len(append))
                return pd.DataFrame({'seqs': prepend + append, 'quals': qual_empty },columns=['seqs', 'quals'], index=self.index)
            else:
                return pd.DataFrame(prepend + append,columns=['seqs'], index=self.index)
        
        substring = pd.DataFrame({name:self.seq_table.loc[:, positions].values.copy().view('S{0}'.format(num_chars)).ravel()}, index=self.index)
        
        if prepend or append:
            substring['seqs'] = substring['seqs'].apply(lambda x: prepend + x + append)

        if not self.qual_table is None and return_quality:
            subquality = self.qual_table.loc[:, positions].values
            subquality = (subquality + self.phred_adjust).copy().view('S{0}'.format(num_chars)).ravel()
            substring['quals'] = subquality
            if prepend or append:
                prepend = '!' * len(prepend)
                append = '!' * len(append)
                substring['quals'] = substring['quals'].apply(lambda x: prepend + x + append)

        return substring

    def get_seq_dist(self, positions=None):
        """
            Returns the distribution of bases or amino acids at each position.
        """
        compare = self.seq_table.loc[:, positions] if positions else self.seq_table
        dist = compare.apply(pd.value_counts).fillna(0)
        dist.rename({c: chr(c) for c in list(dist.index)}, inplace=True)
        return dist

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

    def seq_logo(self):
        pass

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
        from collections import OrderedDict
        if bins is None:
            # use default bins as defined by fastqc report
            bins = [
                1, 2, 3, 4, 5, 6, 7, 8, 9,
                (10, 14), (15, 19), (20, 24), (25, 29), (30, 34), (35, 39), (40, 44), (45, 49), (50, 54), (55, 59), (60, 64),
                (65, 69), (70, 74), (80, 84), (85, 89), (90, 94), (95, 99),
                (100, 104), (105, 109), (110, 114), (115, 119), (120, 124), (125, 129), (130, 134), (135, 139), (140, 144), (145, 149), (150, 154), (155, 159), (160, 164), (165, 169), (170, 174), (175, 179), (180, 184), (185, 189), (190, 194), (195, 199),
                (200, 204), (205, 209), (210, 214), (215, 219), (220, 224), (225, 229), (230, 234), (235, 239), (240, 244), (245, 249), (250, 254), (255, 259), (260, 264), (265, 269), (270, 274), (275, 279), (280, 284), (285, 289), (290, 294), (295, 299),
                300
            ]
            bins = [x if isinstance(x, int) else (x[0], x[1]) for x in bins]
        else:
            # just in case its a generator (i.e. range function)
            bins = [x for x in bins]

        binnames = OrderedDict()
        for b in bins:
            if isinstance(b, int):
                binnames[str(b)] = (b, b)
            elif len(b) == 2:
                binnames[str(b[0]) + '-' + str(b[1])] = (b[0], b[1])

        def get_binned_cols(column):
            # use this function to group together columns
            for n, v in binnames.items():
                if column >= v[0] and column <= v[1]:
                    return n

        temp = self.qual_table.replace(0, np.nan) if exclude_null_quality else self.qual_table
        if sample:
            temp = temp.sample(sample)

        def agg_fxn(group, per):
            # use this function to aggregate quality scores and create distributions/boxplots
            col = group.columns
            name = str(col[0]) if len(col) == 1 else str(col[0]) + '-' + str(col[-1])
            per = [round(p, 2) for p in per]
            to_add_manually = set([0.10, 0.25, 0.50, 0.75, 0.90]) - set(per)
            program_added_values = {f: str(int(f * 100)) + '%' for f in to_add_manually}
            per = per + list(to_add_manually)
            g = group.stack().describe(percentiles=per)
            # Now create a small fake distribution for making box plots. This is preferred over just storing all millions of datapoint
            l = 100
            # man this is ugly, gotta clean this up some hwow
            storevals = [g.loc['min'], g.loc['10%'], g.loc['25%'], g.loc['50%'], g.loc['75%'], g.loc['90%'], g.loc['max']]
            if g.loc['50%'] < 20:
                color = 'red'
            elif g.loc['50%'] < 30:
                color = 'blue'
            else:
                color = 'green'
            subsets = [int(x) for x in np.arange(0, 1, 0.05) * l]
            sample_data = np.zeros(l)
            sample_data[0:subsets[1]] = storevals[1]
            sample_data[subsets[1]:subsets[3]] = storevals[1]
            sample_data[subsets[3]:subsets[7]] = storevals[2]
            sample_data[subsets[7]:subsets[13]] = storevals[3]
            sample_data[subsets[13]:subsets[17]] = storevals[4]
            sample_data[subsets[17]:subsets[19]] = storevals[5]
            sample_data[subsets[19]:] = storevals[5]
            median = g.loc['50%']
            # now only store the values the user wanted
            g = g.drop(program_added_values.values())
            return (g, go.Box(
                    y=sample_data,
                    pointpos=0,
                    name=name,
                    boxpoints=False,
                    fillcolor=color,
                    showlegend=False,
                    line={
                        'color': 'black',
                        'width': 0.7
                    },
                    marker=dict(
                        color='rgb(107, 174, 214)',
                        size=3
                    ),
                    ), median
                    )

        # group results using the aggregation function defined above
        grouped_data = temp.groupby(get_binned_cols, axis=1).apply(lambda groups: agg_fxn(groups, percentiles))
        
        labels = [b for b in binnames.keys() if b in grouped_data.keys()]
        data = pd.DataFrame(grouped_data.apply(lambda x: x[0])).transpose()[labels]

        graphs = list(grouped_data.apply(lambda x: x[1]).transpose()[labels])
        # median_vals = list(grouped_data.apply(lambda x: x[2]).transpose()[labels])
        scatter_min = go.Scatter(x=data.columns, y=data.loc['min'], mode='markers', name='min', showlegend=False)
        # scatter_median = go.Scatter(x=data.columns, y=median_vals, mode='line', name='median', line=dict(shape='spline')
        scatter_mean = go.Scatter(x=data.columns, y=data.loc['mean'], mode='line', name='mean', line=dict(shape='spline'))
        graphs.extend([scatter_min, scatter_mean])
        return data, graphs


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
