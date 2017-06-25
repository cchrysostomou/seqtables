import numpy as np
from .seq_table_util import degen_to_base

"""
Methods for generating a set of fake sequences
"""


def generate_sequence(seq_len=100, chars='ACTG', p_bases=[0.25, 0.25, 0.25, 0.25]):
    """
    Create a random DNA sequence

    Args:
        seq_len (int): Total characters in sequence. default = 100
        chars (str): Each character in string represents a base allowed in sequence
        p_bases (nparray/list floats): probablity for each letter
    Returns:
        seq (str): Sequence generated
    """
    assert len(chars) == len(p_bases)

    seq_len_str = str(seq_len)

    if not isinstance(chars, list):
        chars = list(chars)

    return str(np.random.choice(chars, (seq_len,), p=p_bases).astype('U1').view('U' + seq_len_str)[0])


def generate_library(
    scaffold_seq, num_seqs, error_prone_rate=0, no_error_prone_pos=[], ss_pos=[],
    site_saturation={}, default_site_saturation='N', return_as='seq'
):
    """
    Create a fake library of DNA sequences using a scaffold sequence

    Args:
        scaffold_seq (str): Sequence to create a library from (starting wildtype sequence)
        num_seqs (int): Number of sequences to generate
        error_prone_rate (float): Error prone rate (will assume a poisson distribution)
        no_error_prone_pos (list of ints): Columns/base positions that should NOT undergo mutation
        ss_pos (list_of_ints): Columns/base positions that SHOULD be site saturated
        site_saturation (dict): Each key should be an integer corresponding to a base position defined in ss_pos. Each value should be a either

            1) A character corresponding to degenerate bases that are allowed at that position
            2) A two element tuple corresponding to (letter, probability of selection)

        default_site_saturation (char): Letter defining the default degenerate base distribution to use for a SS position
        return_as (allowed values = 'let', 'seq'): Return values as an nparray of characters, or return as a np array of full length sequences

    Returns:
        list of seqs


    ..note:: Order of operations

        If defining both an error prone event and a site saturation event at the same position, site saturation will occur first, then an error prone

    ..note:: base positions

        Function assumes that bases start at 1 and not 0 (i.e. not python indexing)
    """

    # convert positions to indices
    no_error_prone_pos = [b - 1 for b in no_error_prone_pos]

    # make sure all site saturated positions are included
    ss_pos = sorted(ss_pos + list(site_saturation.keys()))

    # generate sequences
    seq_as_array = np.array([scaffold_seq]).astype('S').view('S1')
    seq_list = np.tile(seq_as_array, num_seqs).reshape(num_seqs, -1)

    site_saturation = {p: site_saturation[p] if p in site_saturation.copy() else default_site_saturation for p in ss_pos}

    degen_to_base_rev = {v: b for b, v in degen_to_base.items()}
    for l in ['A', 'C', 'T', 'G']:
        degen_to_base_rev[l] = l

    # perform site-saturation mutagenesis
    for p in ss_pos:
        ind = p - 1

        # determine how to saturate bases at this position
        if isinstance(site_saturation[p], str):
            assert site_saturation[p] in list(degen_to_base_rev.keys())
            allowed_lets = degen_to_base_rev[site_saturation[p]]
            probs = [1.0 / len(allowed_lets)] * len(allowed_lets)
            lets = list(allowed_lets)
        elif isinstance(site_saturation[p], list):
            lets = [l[0] for l in site_saturation[p]]
            probs = np.array([l[1] for l in site_saturation[p]])
            probs = (probs * 1.0) / probs.sum()
        else:
            raise Exception('Error: invalid format for site_saturation')

        # randomly choose bases
        seq_list[:, ind] = np.random.choice(lets, (num_seqs,), p=probs).astype('S1')

    # # perform error-prone mutagenesis
    ep_pos = sorted(list(set(range(len(scaffold_seq))) - set(no_error_prone_pos)))
    # slice columns/positions we will mutate
    can_mutate = seq_list[:, ep_pos].ravel()
    # randomly select positions to be mutated
    mutate_these_pos = np.random.choice([False, True], can_mutate.shape, p=[1.0 - error_prone_rate, error_prone_rate]).ravel()
    total_mutations = mutate_these_pos.sum()
    # for mutated positions, randomly choose from ACTG
    new_bases = np.random.choice(list('ACTG'), (total_mutations,)).astype('S1')
    # update seqs and reshape to original size
    can_mutate[mutate_these_pos] = new_bases
    can_mutate = can_mutate.reshape(num_seqs, len(ep_pos))
    # update seq list with mutations
    seq_list[:, ep_pos] = can_mutate
    del mutate_these_pos, can_mutate, new_bases

    if return_as == 'seq':
        # return full length sequence as an array
        return seq_list.view('S' + str(seq_list.shape[1])).squeeze()
    elif return_as == 'let':
        # maintain view as a table of seq/pos
        return seq_list
    else:
        raise Exception('Invalid option for return_as parameter. only allow "seq" or "let"')


def add_quality_scores(
    sequence_list, read_type='r1', min_quality=0, max_quality=40,
    starting_mean_quality=36, ending_mean_quality=15, stdV=5, phred_adjust=33, bulk_size=None
):
    """
        Adds quality scores with a moving mean as a function of distance from start of sequencing

        Args:
            sequence_list (np array (n x 1 matrix  or n x b)):  rows (n) represent number of sequences, columns (b) represent number of bases in sequence
            read_type ('r1' or 'r2'): Is the sequence an r1 read or an r2 read (i.e. does sequence start at 5' or 3')
            min_quality (int): minimum allowed quality in a read
            max_quality (int): maximum allowed quality in a read
            starting_mean_quality (mean value of the read quality in the first base)
            ending_mean_quality (mean value of the read quality in the last base)
            stdV (float OR function): determines how to calculate the standard deviation of sequences. If float then it will assume uniform standard deviation.
                If a function then it will assume function takes in base position and returns a float

                    .. note:: Example for std as a function

                    ```
                        # Have the standard deviation increase linearly as a function of base position
                        stdV=lambda(b): 2*b + 1.0
                    ```

            phred_adjust (int): character to associate with a quality score of 0
            bulk_size (int): Bulk size to use when generating random quality scores (i.e. if we dont want to generate 1000000 qualities at the same time creating large
            memory requiremnts, we can set bulk_size to 1000 and only generate 1000 qualtiy scores at a time)


        Returns:
            np array (base quality scores for each read)
    """
    # guess format of sequences provided
    if isinstance(sequence_list, list):
        sequence_list = np.array(sequence_list)
    if len(sequence_list.shape) == 1 or sequence_list.shape[1] == 1:
        # len shape == 1 => does not have a 2nd dimension
        return_as = 'seqs'
        # assume they provided only sequences and not a matrix of sxb, so, will need to calculate sequences
        max_seq_len = np.apply_along_axis(arr=sequence_list.reshape(-1, 1), func1d=lambda x: len(x[0]), axis=1).max()
    else:
        return_as = 'let'
        # assume let are represented by columns
        max_seq_len = sequence_list.shape[1]

    # create a normal distribution with mean 0, and std 1
    if bulk_size is None:
        bulk_size = int(sequence_list.shape[0] * max_seq_len)

    qualities = [
        # create random values of given bulk size (convert to integer afterwards to minimize memory)
        np.random.randn(
            min(bulk_size, (sequence_list.shape[0] * max_seq_len) - ind),
            1
        ).astype(np.uint8)
        for ind in range(0, sequence_list.shape[0] * max_seq_len, bulk_size)
    ]

    qualities = np.stack(qualities).reshape(sequence_list.shape[0], max_seq_len)

    # calculate the mean at each base position
    # use a log distribution to create a slowly decreasing curve
    # for an r2 read... a * log(1 + b) = quality, where quality @ base 0 = ending_mean_quality, @(#bases) = starting_mean_quality
    # r2 read...
    # a * log(0 + 1) + b = ending_mean_quality
    # a * log(max_seq_len + 1) + b = starting_mean_quality
    # b = ending_mean_quality, a = (ending-starting)/(log(1.0/(1.0 + max_seq_len)))
    b, a = ending_mean_quality, (ending_mean_quality - starting_mean_quality) / np.log(1.0 / (1.0 + max_seq_len))
    mean_qualities = (a * np.log(np.arange(0, max_seq_len) + 1.0) + b).astype(np.uint8)

    if read_type == 'r1':
        # r1 should have a flipped version of calculated mean qualities (should start high and go low)
        mean_qualities = mean_qualities[::-1]
    elif read_type == 'r2':
        mean_qualities = mean_qualities
    else:
        raise Exception('Invalid read type: ' + read_type)

    if callable(stdV):
        # calculate standard deviation as a function of base position
        std_vals = stdV(np.arange(0, max_seq_len)).astype(np.uint8)
    else:
        # standard deviation is constant
        std_vals = np.array([stdV] * max_seq_len).astype(np.uint8)

    # add in the mean values and standardeviation for quality at each position
    qualities = qualities * std_vals.reshape(1, -1) + mean_qualities.reshape(1, -1)
    qualities[qualities < min_quality] = int(min_quality)
    qualities[qualities > max_quality] = int(max_quality)

    qualities = qualities.round().astype(np.uint8)

    if return_as == 'let':
        return (qualities + phred_adjust).view('S1')
    else:
        return (qualities + phred_adjust).view('S' + str(max_seq_len)).squeeze()


def randomly_add_indels(
    sequence_list, qual_list=None, insertion_rate=0.001, deletion_rate=0.1, expected_cons_ins = 1,
):
    lets = np.array(sequence_list).astype('S').view('S1')
    delete_these_pos = np.random.choice([False, True], lets.shape, p=[1.0 - deletion_rate, deletion_rate]).ravel()
    lets[delete_these_pos] = '-'
    if qual_list:
        quals = np.array(qual_list).astype('S').view('S1')
        quals[delete_these_pos] = ' '
    if insertion_rate:
        pass