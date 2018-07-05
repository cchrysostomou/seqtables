from seqtables import SeqTable
import pandas as pd
from seqtables.__init__ import bio_installed, SeqIO
import gc

"""
methods for converting files from NGS into seqtables
"""


def read_fastq(input_file, limit=None, chunk_size=10000, use_header_as_index=True, use_pandas=True, ignore_quotes=True):
    """
        Load a fastq file as class SeqTable
    """
    def group_fastq(index):
        return index % 4

    # limit refers to reads, fastq file is 4 lines per read
    line_limit = limit * 4 if limit else None

    if use_pandas:
        header = []
        seqs = []
        quals = []

        if ignore_quotes:
            if chunk_size is not None:
                dfs = pd.read_csv(input_file, sep='\n', nrows=line_limit, chunksize=chunk_size, quotechar='\x07', header=None)
            else:
                dfs = [pd.read_csv(input_file, sep='\n', nrows=line_limit, quotechar='\x07', header=None)]
        else:
            if chunk_size is not None:
                dfs = pd.read_csv(input_file, sep='\n', nrows=line_limit, chunksize=chunk_size, header=None)
            else:
                dfs = [pd.read_csv(input_file, sep='\n', nrows=line_limit, header=None)]

        for tmp in dfs:
            tmp = tmp.groupby(group_fastq)
            header.extend(list(tmp.get_group(0)[0].apply(lambda x: x[1:].strip())))
            seqs.extend(list(tmp.get_group(1)[0].apply(lambda x: x.strip())))
            quals.extend(list(tmp.get_group(3)[0].apply(lambda x: x.strip())))
    else:
        if bio_installed is False:
            raise Exception("You do not have BIOPYTHON installed and therefore must use pandas to read the fastq (set use_pandas parameter to True). If you would like to use biopython then please install")
        seqs = []
        quals = []
        header = []
        l = 0
        for seq in SeqIO.parse(input_file, 'fastq'):
            r = seq.format('fastq').split('\n')
            header.append(r[0])
            seqs.append(r[1])
            quals.append(r[3])
            if limit is not None and l > limit:
                break
    st = SeqTable(seqs, quals, index=header, seqtype='NT')
    del seqs, quals, header
    gc.collect()
    return st


# def read_sam(input_file, limit=None, chunk_size=100000, cleave_softclip=False, use_header_as_index=True, ignore_quotes=True):
#     """
#         Load a SAM file into class SeqTable
#     """
#     skiplines = 0
#     with open(input_file) as r:
#         for i in r:
#             if i[0] == '@':
#                 skiplines += 1
#             else:
#                 break

#     cols_to_use = [9, 10]
#     if use_header_as_index:
#         cols_to_use.append(0)
#         index_col = 0
#     else:
#         index_col = None

#     if cleave_softclip:
#         cols_to_use.append(5)

#     df = pd.read_csv(input_file, sep='\t', header=None, index_col=index_col, usecols=cols_to_use, skiprows=skiplines, quotechar='\x07') if ignore_quotes else pd.read_csv(input_file, sep='\t', header=None, index_col=index_col, usecols=cols_to_use, skiprows=skiplines)

#     index = df.index
#     return SeqTable(df, index=index, seq_type='NT')
