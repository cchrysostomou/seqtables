import pandas as pd
import re
import numpy as np
from collections import OrderedDict, defaultdict
import time
import warnings
try:
    import regex
    regex_found = True
except ImportError as e:
    # warnings.warn('Warning, module regex not found . please install for expected functionality')
    import re as regex
    regex_found = False

from seqtables.core.utils.alphabets import extended_cigar_alphabet

def breakdown_bits(flag, asstring=True):
    """
    1 0x1 template having multiple segments in sequencing
    2 0x2 each segment properly aligned according to the aligner
    4 0x4 segment unmapped
    8 0x8 next segment in the template unmapped
    16 0x10 SEQ being reverse complemented
    32 0x20 SEQ of the next segment in the template being reverse complemented
    64 0x40 the first segment in the template
    128 0x80 the last segment in the template
    256 0x100 secondary alignment
    512 0x200 not passing filters, such as platform/vendor quality controls
    1024 0x400 PCR or optical duplicate
    2048 0x800 supplementary alignment
    """
    
    if flag == 0:
        return [] if asstring is True else ''
    flags = [n for n in range(int(np.log2(flag)) + 1) if flag & (1 << n)]
    if asstring:
        flags = ','.join([str(x) for x in flags])
    return flags


def return_read_num(flag):
    """
    Determine whether read is R1 or R2. Assumes that R1 is forward direction and R2 is reverse direction
    """
    index = np.log2(16)
    if flag == 0:
        return 'R1'
    return 'R2' if index in breakdown_bits(flag, False) else 'R1'


def filter_flags(flag, settings):
    return len(set(breakdown_bits(flag, False)) & set(settings)) == 0


def filter_flags_read_num(flag, settings):
    if len(set(breakdown_bits(flag, False)) & set(settings)) == 0:
        index = np.log2(16)
        return 'R2' if index in breakdown_bits(flag, False) else 'R1'
    else:
        return np.nan


def cigar_breakdown(cigarstring):
    """
        Breaks down a cigar string into all reported events (softclipping, insertions, deletions, matches)

        Parameters
        ----------
        cigarstring: cigar string from an aligned sequence

        Returns
        -------
        ordered_mutations (list of tuples): an ordered list of all events (type, # of bases)
        mutation_summary (dict): a dict showing the number of times events were detected (i.e. total matches, total deletions)
    """
    assert regex_found is True, 'Please install regex module before attempting to read samfiles'
    re_search_string = r'((\d+)([{0}]))+'.format(extended_cigar_alphabet)
    result = regex.match(re_search_string, cigarstring)
    if result:
        ordered_mutations = [(m, int(n)) for m, n in zip(result.captures(3), result.captures(2))]
        mutation_summary = defaultdict(int)
        for (m, n) in ordered_mutations:
            mutation_summary[m] += n
        return ordered_mutations, mutation_summary
    else:
        return [], {}


def get_nterminal_softclip(cigarstring):
    """
        Extracts the N-terminal softclipping parameter from a cigar string. Reports softclipping as an integer.

        Args:
            cigarstring (str): cigar string from an aligned sequence

        Returns:
            five_clipping (int): the number of bases from 5' of  raw sequence that was stripped for alignment
    """
    five_clipping = re.search(r'^(\d+)(?=S)', cigarstring)
    return int(five_clipping.groups()[0]) if five_clipping else 0


def get_cterminal_softclip(cigarstring):
    """
        Extracts the C-terminal softclipping parameter from a cigar string. Reports softclipping as an integer.

        Parameters
        ----------
        cigarstring: cigar string from an aligned sequence

        Returns
        -------
        three_clipping : the number of bases from 3' of  raw sequence that was stripped for alignment
    """
    three_clipping = re.search(r'(\d+)S$', cigarstring)
    return int(three_clipping.groups()[0]) if three_clipping else 0


def get_base_hits(cigarstring):
    matches = re.search(r'(\d+)M', cigarstring)
    return int(matches.groups()[0]) if matches else 0


def make_seq_algn(row, max_l):
    if row['cterminal_clip'] == 0:
        endp = len(row['seq'])
    else:
        endp = -row['cterminal_clip']
    seq_substr = row['seq'][row['nterminal_clip']:endp]
    qual_substr = row['qual'][row['nterminal_clip']:endp]
    prefixN = 'N' * (row['start'] - 1)
    endN = 'N' * (max_l - ((row['start'] - 1) + len(seq_substr)))
    prefixQ = chr(33) * (row['start'] - 1)
    endQ = chr(33) * (max_l - ((row['start'] - 1) + len(seq_substr)))
    return {'seq_algn': prefixN + seq_substr + endN, 'qual_algn': prefixQ + qual_substr + endQ}


def filter_reads(df, ignore_hits=[], phix_filter=True, remove_indels=True, bits_not_allowed=[4, 512], expected_start={}, expected_end={}, debug_file=None):
    """
    Filter aligned reads from a dataframe generated from a SAM file
    Args:
        df (dataframe): data frame generated from read_sam function
        phix_filter (boolean): remove reads with a hit from phix
        bits_not_allowed (list of ints): remove reads containing any of these bits in the flag field
        expected_start (dict): If defined, remove reads that do not start at the expected alignment position
        expected_end (dict): If defined, remove reads that do not end at the expected alignment position
        debug_file (string): location of a debug file to generate while parsing

    Returns:
        df (dataframe): filtered dataframe based on rules
        baddf (dataframe): results which were filtered form dataframe with reason why
        stats (dict): a counter of all reads that were filtered and why
    """

    if not isinstance(ignore_hits, list):
        ignore_hits = [ignore_hits]
    filter_hits = ['', '*'].extend(ignore_hits) if ignore_hits else ['', '*']
    if 'error' not in df.columns:
        df['error'] = ''
    stats = OrderedDict()
    stats['pre_filter_reads'] = df.shape[0]
    dflen = df.shape[0]
    dftemps = []
    bad_hits = df['rname'].isin(filter_hits)

    df.loc[bad_hits, 'error'] = 'invalid hits'
    # tmp = df[df['rname'].isin(filter_hits)]
    # tmp['error'] = 'invalid hits'
    # dftemps.append(tmp)
    filtervals = df.loc[bad_hits]
    df.drop(filtervals.index, inplace=True)
    dftemps.append(filtervals)

    # df = df[~df['rname'].isin(filter_hits)]
    stats['error: invalid hits'] = filtervals.shape[0]
    dflen = df.shape[0]
    if phix_filter:
        phix_hit = df.rname.str.lower().str.startswith('phix')
        df.loc[phix_hit, 'error'] = 'phix'
        filtervals = df.loc[phix_hit]
        df.drop(filtervals.index, inplace=True)
        dftemps.append(filtervals)

    stats['error: phix'] = dflen - df.shape[0]
    dflen = df.shape[0]
    if bits_not_allowed:
        bits_not_allowed = [np.log2(x) for x in bits_not_allowed]
        df['read'] = df['flag'].apply(filter_flags_read_num, settings=bits_not_allowed)
        df.loc[df['read'].isnull(), 'error'] = 'invalid SAM flags'
        tmp = df[df['read'].isnull()]
        dftemps.append(tmp)
        df.drop(tmp.index, inplace=True)
    else:
        df['read'] = df['flag'].apply(return_read_num)

    stats['error: invalid SAM flags'] = dflen - df.shape[0]
    dflen = df.shape[0]

    if remove_indels is True:
        contains_index = df['cigar'].str.contains('[ID]')
        df.loc[contains_index, 'error'] = 'contains indels'
        filtervals = df.loc[contains_index]
        dftemps.append(filtervals)
        df.drop(filtervals.index, inplace=True)
        stats['error: contains indels'] = filtervals.shape[0]

    unaccounted_for_strings = df['cigar'].str.contains('[HP=XN]')
    if debug_file:
        df.loc[unaccounted_for_strings].to_csv(debug_file, sep='\t')
    df.loc[unaccounted_for_strings, 'error'] = 'unaccounted cigar parameters'
    filtervals = df.loc[unaccounted_for_strings]
    dftemps.append(filtervals)
    df.drop(filtervals.index, inplace=True)
    stats['error: unaccounted cigar parameters'] = filtervals.shape[0]
    df['nterminal_clip'] = ''
    df['cterminal_clip'] = ''

    df[['nterminal_clip', 'cterminal_clip']] = df.cigar.str.extract(r'^(\d+(?=S))?.+?(\d+)?S?$', expand=True).fillna(0).astype(int).values  # = df['cigar'].str.extract().apply(get_nterminal_softclip)
    # df['cterminal_clip'] = df['cigar'].apply(get_cterminal_softclip)
    stats['post_filter_reads'] = dflen
    df.sort_values(by=['header', 'read'], inplace=True)
    df['pos_end'] = df['pos'] + df['seq'].apply(len) - df['nterminal_clip'] - df['cterminal_clip'] - 1
    baddf = pd.concat(dftemps)

    return df, baddf, stats


def read_sam(file, std_fields_keep=['header', 'flag', 'rname', 'pos', 'cigar', 'seq', 'qual'], opt_fields_keep=['XN', 'XM', 'MD'], nrows=None, chunks=None, indexing_dict = None, ignore_quotes=True, comment_out_letter=False):
    """
        Read a sam file for the first time and process the fields into a dataframe

        Args:
            file (string or tuple): location of sam file. If instance of file is a tuple, then we assume that it is of the following format (filepath, file-label or index lable)

                .. note::Labeling

                    You can label all the reads in this file using a tuple to define both the filename and label. This could be useful when you want to merge multiple dataframes from multiple
                    sources/indexes

                    >>> read_sam((path, 'label1'))

            std_fields_keep (list of strings): fields we want to keep from the SAM file
            opt_fields_keep (list of strings): fields from the optional section we want to keep
            nrows (int, default None): If defined only read in this many rows
            chunks (int, default None): If defined, only read in SAM file this many lines at a time. If not defined, loads entire file into memory
            index_label (list of tuples, default=None): If defined, adds a new column to the dataframe for seperating reads into 'indexes/labels' by the filename
            ignore_quotes (boolean): Pandas will treat quotes as literal. So it will not include '"' quotes in the string. This can be an issue when reading in quality scores. If True it will force pandas to include quoted characters in the string.
    """
    def extract_optional_features(row):
        """
        Parses optional fields from SAM file into columns
        """
        # optional fields:
        # 'AS' = ALIGNMENT SCORE FOR BEST ALIGNMENT
        # 'XS' = ALIGNMENT SCORE FOR 2ND BEST ALIGNMENT
        # 'YS' = ALIGNMENT SCORE FOR OPPOSIT EMATE IN PAIRED END ALIGNMENT
        # 'XN' = # OF AMBIGUOOUS BASES IN THE REFERENCE COVERING THIS ALIGNMENT
        # 'XM' = # OF MISMATCHED IN ALIGNMENT
        # 'XO' = # NUMBER OF GAP OPINS FOR BOTH READ AND REVERFERCNE GAPS
        # 'XG' = # NUMBER OF GAP EXTENSIONS FOR BOTH READ AND REFERNCE GAPS
        # 'NM' = EDIT DISTANCE, NUMBER OF ONE NUCLEOTIDE EDITS (SUB, INS, DELETE) NEEDED TO TRANSFORM THE READ STRING INTO REFERENCE STRING
        # 'YP' = EQUALS 1 IF THE READ IS PART OF A PAIR THA THAS AT LEAST N CONDORDINATE ALIGNMENTS WHER EN IS ARGUMENT SPECIFIED TO M PLUST 0
        # 'YM' = EQUALS 1 IF THE READ ALIGNED WITH AT LEAST N UNPAIRED ALIGNMENTS WHERE IS THE ARGUMENT SPECIFIED TO -M + 1
        # 'YF' = STRING INDICIATING WHY READ WAS FILTERED OUT
        # 'MD:Z' = A STRING REPRESENTATION OF THE MISTAMCHED REFERNCE BASES IN THE ALIGNMENT
        split_vals = [val.split(':') for val in row if not pd.isnull(val)]
        new_fields = {v[0]: v[-1] for v in split_vals}
        return new_fields

    # if isinstance(file, tuple):
    #     # add a unique identifier to all reads in this file
    #     index_label = file[1]
    #     file = file[0]
    # else:
    #     index_label = None

    sam_column_names = ['header', 'flag', 'rname', 'pos', 'mapq', 'cigar', 'rnext', 'pnext', 'tlen', 'seq', 'qual']
    if not std_fields_keep:
        std_fields_keep = sam_column_names

    quoted_char = '\x07' if ignore_quotes else '"'

    # first check for comment lines at beginning
    num_skip = 0
    total_cols = 0

    with open(file) as sb:
        for line in sb:
            if line.startswith('@'):
                num_skip += 1
            else:
                total_cols = len(line.split('\t'))
                break

    # opt_fields_keep = ['XN', 'XM', 'MD']
    num_optional_fields = total_cols - len(sam_column_names)
    opt_fields = ["opt_field_" + str(n + 1) for n in range(num_optional_fields)]
    cout = '@' if comment_out_letter else None  # adding comment='@' does not work because of quality scores (they get commented out!)
    if chunks:
        # lets make sure we are always adding chunks in in "pairs". this way we can hardcode our read_sam and assume that R1-R2 pairs are always following one another...
        # Will this break?
        chunks = 2 * (chunks / 2)
        dfgen = pd.read_csv(file, sep='\t', comment=cout, skiprows=num_skip, header=None, names=sam_column_names + opt_fields, chunksize=chunks, engine='c', quotechar=quoted_char)
    else:
        dfgen = [pd.read_csv(file, sep='\t', comment=cout, skiprows=num_skip, header=None, names=sam_column_names + opt_fields, nrows=nrows, engine='c', quotechar=quoted_char)]
    # print(sam_column_names + opt_fields)
    total_reads = 0
    for df in dfgen:
        if total_cols == 0:
            yield pd.DataFrame([])
            continue
        # print(df.iloc[0])
        total_reads += df.shape[0]
        optional_columns = [c for c in df.columns if c.startswith('opt_field')]

        if optional_columns:
            if opt_fields_keep == []:
                # NOTHING TO KEEP SO DROP ALL OPTIONAL FIELDS
                # print(df.iloc[0])
                df = df.drop(optional_columns, axis=1)
                # print(df.iloc[0])
            else:
                # convert the optional fields into columns (i.e. MD:Z:VALUE...)
                # print(opt_fields_keep)
                # print(optional_columns)
                # print(df.iloc[0])
                # print(df[optional_columns].fillna("").apply(extract_optional_features, axis=1, raw=True)/)
                # print(df[optional_columns].head())
                # print(optional_columns)
                keys = []
                # print(df[optional_columns].fillna("").apply(extract_optional_features, axis=1, raw=True))
                for x in df[optional_columns].fillna("").apply(extract_optional_features, axis=1, raw=True):
                    keys.extend(x.keys())
                # print(sorted(list(set(keys))))
                # print('x')
                # tmp = df[optional_columns].fillna("").apply(extract_optional_features, axis=1, raw=True)[:10]
                # print(tmp[0])
                new_columns = pd.DataFrame.from_records(df[optional_columns].fillna("").apply(extract_optional_features, axis=1, raw=True)[:10])
                new_columns = pd.DataFrame.from_records(df[optional_columns].fillna("").apply(extract_optional_features, axis=1, raw=True))
                opt_fields_keep = [o for o in opt_fields_keep if o in new_columns.columns]

                # update results
                if not new_columns.empty:
                    df[opt_fields_keep] = new_columns[opt_fields_keep]
        keep_columns = std_fields_keep + opt_fields_keep
        df = df[keep_columns]

        if nrows and nrows < total_reads:
            reads_required = df.shape[0] - (total_reads - nrows)
            if reads_required > 0:
                yield df.iloc[:(df.shape[0] - (total_reads - nrows)), :]
            break

        if indexing_dict is not None:
            df['indexing'] = indexing_dict[file]

        # else:
        #     df['indexing'] = ''
        # if index_label is not None:
        #    df['indexing'] = index_label
        yield df
        del df
