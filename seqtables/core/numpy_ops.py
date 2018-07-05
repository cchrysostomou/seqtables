import numpy as np
import pandas as pd
import xarray as xr
import copy
import warnings

try:
    from plotly import graph_objs as go
    plotly_installed = True
except:
    plotly_installed = False
    # warnings.warn("PLOTLY not installed so interactive plots are not available. This may result in unexpected funtionality")

global_3d_mapper = np.repeat(0, 256 * 4).reshape(256, -1)
global_3d_mapper[ord('T'), :] = np.array([0, 0, 0, 1])
global_3d_mapper[ord('C'), :] = np.array([0, 1, 0, 0])
global_3d_mapper[ord('A'), :] = np.array([1, 0, 0, 0])
global_3d_mapper[ord('G'), :] = np.array([0, 0, 1, 0])


def compare_sequence_matrices(seq_arr1, seq_arr2, flip=False, treat_as_match=[], ignore_characters=[], return_num_bases=False):
    """
        This will "align" seq_arr1 to seq_arr2. It will calculate which positions in each sequence defined by seq_arr1 matches each position in each sequence defined by seq_arr2

        seq_arr1 = NxP matrix where N = # of sequences represented in seq_arr1 and P represents each base pair position/the length of the string
        seq_arr2 = MxP matrix where M = # of sequences represented in seq_arr1 and P represents each base pair position/the length of the string

        This operation will return a NxPxM boolean matrix where each position represents whether the base pair in sequence N and the base pair in sequence M represented at position P match
        In other words, if bool_arr = compare_sequence_matrices(A, B) then the total hamming distance between the second and third sequence in matrices A and B respective can be found as

        >>> bool_arr.sum(axis=1)[1][2]

        Args:
            seq_arr1 (np.array): MxP matrix of sequences represented as array of numbers
            seq_arr2 (np.array): NxP matrix of sequences represented as array of numbers
            flip (bool): If False then "true" means that letters are equal at specified positoin, If True then return positions that are NOT equal to one another
            treat_as_match (list of chars): Treat any positions that have any of these letters in either matricies as True
            ignore_characters (list of chars): Ignore positions that have letters in either matricies at specified positions

                .. warning:: datatype

                    When ignore character is defined, the array is passed back as a np.float dtype because it must accomodate np.nan

            return_num_bases (False): If true then it will return a second parameter that defines the number of non nan values between alignments

        Returns: NxPxM array of boolean values
    """

    assert seq_arr1.shape[1] == seq_arr2.shape[1], 'Matrices do not match!'

    # use np.int8 because it ends upbeing faster
    seq_arr1 = seq_arr1.view(np.uint8)
    seq_arr2 = seq_arr2.view(np.uint8)

    # this will return true of pos X in seqA and seqB are equal
    diff_arr = (seq_arr1[..., np.newaxis].view(np.uint8) == seq_arr2.T[np.newaxis, ...])
    # print(diff_arr.shape)
    if treat_as_match:
        # treat any of these letters at any positions as true regardles of whether they match in respective pairwise sequences
        if not isinstance(treat_as_match, list):
            treat_as_match = [treat_as_match]

        treat_as_match = [ord(let) for let in treat_as_match]

        # now we have to ignore characters that are equal to specific values
        # return True for any positions that is equal to "treat_as_true"

        ignore_pos = ((seq_arr1 == treat_as_match[0])[..., np.newaxis]) | ((seq_arr2 == treat_as_match[0])[..., np.newaxis].T)
        for chr_p in treat_as_match[1:]:
            ignore_pos = ignore_pos | ((seq_arr1 == chr_p)[..., np.newaxis]) | ((seq_arr2 == chr_p)[..., np.newaxis].T)

        # now adjust boolean results to ignore any positions == treat_as_true
        diff_arr = (diff_arr | ignore_pos)  # if flip is False else (diffs | ignore_pos)

    if flip is False:
        diff_arr = diff_arr  # (~(~diffarr))
    else:
        diff_arr = ~diff_arr  # (~diffarr)
    # print(diff_arr.shape)
    if ignore_characters:
        # do not treat these characters as true OR false
        if not isinstance(ignore_characters, list):
            ignore_characters = [ignore_characters]
        ignore_characters = [ord(let) for let in ignore_characters]

        # now we have to ignore characters that are equal to specific values
        ignore_pos = (seq_arr1 == ignore_characters[0])[..., np.newaxis] | ((seq_arr2 == ignore_characters[0])[..., np.newaxis].T)
        for chr_p in ignore_characters[1:]:
            ignore_pos = ignore_pos | ((seq_arr1 == chr_p)[..., np.newaxis]) | ((seq_arr2 == chr_p)[..., np.newaxis]).T

        diff_arr = diff_arr.astype(np.float)
        diff_arr[ignore_pos] = np.nan

    diff_arr = diff_arr

    if return_num_bases:
            num_bases = np.apply_along_axis(
                arr=diff_arr,
                axis=1,
                func1d=lambda x: len(x[~np.isnan(x)])
            )
            return diff_arr, num_bases
    else:
        return diff_arr


def numpy_value_counts_bin_count(arr, weights=None):
    """
    Use the 'bin count' function in numpy to calculate the unique values in every column of a dataframe
    clocked at about 3-4x faster than pandas_value_counts (df.apply(pd.value_counts))

    Args:
        arr (dataframe, or np array): Should represent rows as sequences and columns as positions. All values should be int
        weights (np array): Should be a list of weights to place on each
    """

    if not isinstance(arr, np.ndarray):
        raise Exception('The provided parameter for arr is not a dataframe or numpy array')

    if len(arr.shape) == 1:
        # its a ONE D array, lets make it two D
        arr = arr.reshape(-1, 1)

    arr = arr.view(np.uint8)
    # returns an array of length equal to the the max value in array + 1. each element represents number of times an integer appeared in array.
    bins = [
        np.bincount(arr[:, x], weights=weights)
        for x in range(arr.shape[1])
    ]

    indices = [np.nonzero(x)[0] for x in bins]  # only look at non zero bins

    series = [pd.Series(y[x], index=x) for (x, y) in zip(indices, bins)]
    return pd.concat(series, axis=1).fillna(0)


def get_quality_dist(
    arr, col_names=None, bins='even', exclude_null_quality=True, sample=None,
    percentiles=[10, 25, 50, 75, 90], stats=['mean', 'median', 'max', 'min'],
    plotly_sampledata_size=20, use_multiindex=True,
):
    """
        Returns the distribution of quality across the given sequence, similar to FASTQC quality seq report.

        Args:
            arr (np.array): a matrix of quality scores where rows represent a sequence and columns represent a position
            col_names (list): column header for the numpy array (either from xarray or pandas)
            bins(list of ints or tuples, or 'fastqc', or 'even'): bins defines how to group together the columns/sequence positions when aggregating the statistics.

                .. note:: bins='fastqc' or 'even'

                    if bins is not a set of numbers and instead one of the two predefined strings ('fastqc' and 'even') then calculation of bins will be defined as follows:

                            1. fastqc: Identical to the bin ranges used by fastqc report
                            2. even: Creates 10 evenly sized bins based on sequence lengths

            percentiles (list of floats, default=[10, 25, 50, 75, 90]): value passed into numpy quantiles function.
            exclude_null_quality (boolean, default=True): do not include quality scores of 0 in the distribution
            sample (int, default=None): If defined, then we will only calculate the distribution on a random subsampled population of sequences
            plotly_sampledata_size (int, default=20): Number of values to store in a sample numpy array used for creating box plots in plotly

                .. note:: min size

                    note the minimum value for a sampledata size is 10

        Returns:
            data (DataFrame): contains the distribution information at every bin (min value, max value, desired precentages and quartiles)
            graphs (plotly object): contains plotly graph objects for generating plots of the data afterwards

        Examples:
            Show the median of the quality at the first ten positions in the sequence

            >>> table = SeqTable(['AAAAAAAAAA', 'AAAAAAAAAC', 'CCCCCCCCCC'], qualitydata=['6AA9-C9--6C', '6AA!1C9BA6C', '6AA!!C9!-6C'])
            >>> box_data, graphs = table.get_quality_dist(bins=range(10), percentiles=[50])

            Now repeat the example from above, except group together all values from the first 5 bases and the next 5 bases
            i.e.  All qualities between positions 0-4 will be grouped together before performing median, and all qualities between 5-9 will be grouped together). Also, return the bottom 10 and upper 90 percentiles in the statsitics

            >>> box_data, graphs = table.get_quality_dist(bins=[(0,4), (5,9)], percentiles=[10, 50, 90])

            We can also plot the results as a series of boxplots using plotly
            >>> from plotly.offline import init_notebook_mode, iplot, plot, iplot_mpl
            # assuming ipython..
            >>> init_notebook_mode()
            >>> plotly.iplot(graphs)
            # using outside of ipython
            >>> plotly.plot(graphs)
    """
    from collections import OrderedDict

    current_stats = ['min', 'max', 'mean', 'median']
    assert set(stats).issubset(set(current_stats)), "The stats provided are not currently supported. We only support {0}".format(','.join(current_stats))

    # current base positions in dataframe
    if col_names is None:
        col_names = np.arange(1, arr.shape[1] + 1)
    else:
        assert len(col_names) == arr.shape[1], 'Column names does not match shape'
    # print(bins)

    if bins is 'fastqc':
        # use default bins as defined by fastqc report
        bins = [
            (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9),
            (10, 14), (15, 19), (20, 24), (25, 29), (30, 34), (35, 39), (40, 44), (45, 49), (50, 54), (55, 59), (60, 64),
            (65, 69), (70, 74), (80, 84), (85, 89), (90, 94), (95, 99),
            (100, 104), (105, 109), (110, 114), (115, 119), (120, 124), (125, 129), (130, 134), (135, 139), (140, 144), (145, 149), (150, 154), (155, 159), (160, 164), (165, 169), (170, 174), (175, 179), (180, 184), (185, 189), (190, 194), (195, 199),
            (200, 204), (205, 209), (210, 214), (215, 219), (220, 224), (225, 229), (230, 234), (235, 239), (240, 244), (245, 249), (250, 254), (255, 259), (260, 264), (265, 269), (270, 274), (275, 279), (280, 284), (285, 289), (290, 294), (295, 299),
        ] + [(p, p + 9) for p in np.arange(300, arr.shape[1], 10)]
        bins = [x if isinstance(x, int) else (x[0], x[1]) for x in bins]
    elif bins is 'even':
        # create an equal set of 10 bins based on df shape
        binsize = int(arr.shape[1] / 10)
        bins = []
        for x in range(0, arr.shape[1], binsize):
            c1 = col_names[x]
            c2 = col_names[min(x + binsize - 1, arr.shape[1] - 1)]
            bins.append((c1, c2))
        # print(bins)
    else:
        # just in case its a generator (i.e. range function)
        # convert floats to ints, otherwise keep original
        bins = [(int(x), int(x)) if isinstance(x, float) else x if isinstance(x, tuple) else (x, x) for x in bins]

    binnames = OrderedDict()

    for b in bins:
        if b[0] < min(col_names) or b[0] > max(col_names):
            continue
        # create names for each bin
        if isinstance(b, int):
            binnames[str(b)] = (b, b)
        elif len(b) == 2:
            binnames[str(b[0]) + '-' + str(b[1])] = (b[0], b[1])

    temp = xr.DataArray(
        arr[np.random.choice(arr.shape[0], sample), :] if sample else arr,
        dims=('read', 'position'),
        coords={'position': col_names}
    )

    # define the quantile percentages we will return for each quality bin
    percentiles = [round(p, 0) for p in percentiles]
    per = copy.copy(percentiles)
    # ensure that the following percentiles will ALWAYS be present
    program_required = [0, 10, 25, 50, 75, 90, 100]
    to_add_manually = set(program_required) - set(per)

    # update percentil list
    per = sorted(per + list(to_add_manually))

    # loop through each of the binnames/bin counts
    binned_data = OrderedDict()
    binned_data_stats = OrderedDict()
    graphs = []  # for storing plotly graphs

    plotlychosendata = pd.DataFrame(0, index=list(binnames.keys()), columns=['min', 'max', 'mean', 'median'])

    for name, binned_cols in binnames.items():
        userchosen_stats = {}
        userchosen = {}
        if isinstance(binned_cols, int):
            # not binning together multiple positions in sequence
            binned_cols = (binned_cols, binned_cols)
        # create a list of all column/base positions listed within this bin
        # set_cols = set(list(range(binned_cols[0], binned_cols[1] + 1)))
        # identify columns in dataframe that intersect with columns listed above
        # sel_cols = list(col_names_set & set_cols)
        # select qualities within bin, unwind list into a single list
        p = list(set(np.arange(binned_cols[0], binned_cols[1] + 1)) & set(temp.position.values))  # make sure positions are present in columns
        bin_qual = temp.sel(position=p).values.ravel()
        if exclude_null_quality:
            quantile_res = np.percentile(bin_qual[bin_qual > 0], per)
            mean_val = bin_qual[bin_qual > 0].mean()
            plotlychosendata.loc[name, 'mean'] = mean_val
            if 'mean' in stats:
                userchosen_stats['mean'] = mean_val
        else:
            mean_val = bin_qual[bin_qual > 0].mean()
            quantile_res = np.percentile(bin_qual, per)
            plotlychosendata.loc[name, 'mean'] = mean_val
            if 'mean' in stats:
                userchosen_stats['mean'] = mean_val

        storevals = []
        for p, qnt in zip(per, quantile_res):
            if p == 0:
                plotlychosendata.loc[name, 'min'] = qnt
                if 'min' in stats:
                    userchosen_stats['min'] = qnt
            if p == 100:
                plotlychosendata.loc[name, 'max'] = qnt
                if 'max' in stats:
                    userchosen_stats['max'] = qnt
            if p in program_required:
                # store the values required by the program in storevals
                storevals.append(qnt)
            if p in percentiles:
                # store original quantile values desired by user in variable percentiles
                userchosen[str(int(p)) + '%'] = qnt
            if p == 50:
                # store median
                median = qnt
                if 'median' in stats:
                    userchosen_stats['median'] = qnt
                plotlychosendata.loc[name, 'median'] = qnt

        userchosen = pd.Series(userchosen)

        if plotly_sampledata_size < 10:
            warnings.warn('Warning, the desired plotly_sampledata_size is too low, value has been changed to 10')
            plotly_sampledata_size = 10
        # next a fake set of data that we can pass into plotly for making boxplots. datas descriptive statistics will match current set
        sample_data = np.zeros(plotly_sampledata_size)
        # these indices in subsets indicates the 5% index values for the provided sample_data_size
        subsets = [int(x) for x in np.arange(0, 1.00, 0.05) * plotly_sampledata_size]

        # we hardcoded the values in program_required, so we can add those values into fake subsets
        sample_data[0:subsets[1]] = storevals[1]  # store min value in these indices
        sample_data[subsets[1]:subsets[3]] = storevals[1]  # store bottom 10% of data within 5-15% data range
        sample_data[subsets[3]:subsets[7]] = storevals[2]  # store 25% of data
        sample_data[subsets[7]:subsets[13]] = storevals[3]  # store median of data
        sample_data[subsets[13]:subsets[17]] = storevals[4]  # store 75% of data
        sample_data[subsets[17]:subsets[19]] = storevals[5]  # store max val
        sample_data[subsets[19]:] = storevals[5]  # store max val

        color = 'red' if median < 20 else 'blue' if median < 30 else 'green'

        if plotly_installed is True:
            # create a box plot using the fake sample_data, again this is better for memory resources since plotly stores all datapoints in javascript
            plotdata = go.Box(
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
                )
            )
        else:
            warnings.warn('PLOTLY not installed. No graph object data was returned')
            plotdata = None

        graphs.append(plotdata)
        binned_data[name] = userchosen
        binned_data_stats[name] = userchosen_stats

    if plotly_installed is True:
        # also include a scatter plot for the minimum value, maximum value, and mean in distribution
        scatter_min = go.Scatter(x=list(plotlychosendata.index), y=plotlychosendata['min'], mode='markers', name='min', showlegend=False)
        scatter_max = go.Scatter(x=list(plotlychosendata.index), y=plotlychosendata['max'], mode='markers', name='max')
        scatter_mean = go.Scatter(
            x=list(plotlychosendata.index),
            y=plotlychosendata['mean'], line=dict(shape='spline'),
            name='mean'
        )
        graphs.extend([scatter_min, scatter_max, scatter_mean])

    if use_multiindex is True:
        stats_df = pd.concat([pd.DataFrame(binned_data), pd.DataFrame(binned_data_stats)], keys=['percentile', 'stats'])
    else:
        stats_df = pd.concat([pd.DataFrame(binned_data), pd.DataFrame(binned_data_stats)])

    return stats_df, graphs


def filter_by_count(arr, axis, min_count):
    """
        Filter a numpy array by return values whose unique row/column counts across an axis are > some cutoff
    """
    # first collapse the two levels into their unique values, but also return the inverse of unique values so we can map the original index to its respective unique value
    # is_unique -> represents the mapped index to the position of the unique value in un_vals and un_counts
    # unique_idx => shape = shape of original array => performining un_vals[unique_idx] returns the original array!
    # un_vals => shape of unique values in array
    # un_counts => shape of unique values in array

    un_vals, unique_idx, un_counts = np.unique(
        arr,
        axis=axis,
        return_counts=True,
        return_inverse=True
    )

    # STEPS WE WILL PERFORM
    # first lets represent the respective COUNTS OF EACH UNIQUE VALUE but have its shape = the shape of the original index (this is identical to a "transform" in pandas groupby)
    # count_of_its_unique_value = un_counts[unique_idx]  # (unique_idx is REPETIIVE OF COURSE)

    # # next lets only identify rows that are above a cutoff
    # rows_with_unique_value_above_cutoff = np.where(count_of_its_unique_value > cutoff)

    # # now, since we know which rows (from the unique row table) we are interested in, lets go ahead and actually grab those rows
    # unique_idx[rows_with_unique_value_above_cutoff]

    # # finally, we want to recapitulate the ORIGINAL data from each row defined by unique_idx with uniquevalues > cutoff
    # filtered_rows = un_vals[unique_idx[rows_with_unique_value_above_cutoff]]

    # PUTTING IT ALL TOGETHER WE GET:
    return un_vals[unique_idx[np.where(un_counts[unique_idx] > min_count)]]


def return_3d_arr(arr):
    """
        return a 3D array where:
         * dim1 = # of sequences,
         * dim2 = # of bases per sequence
         * dim3 = a 4 array element for each letter
             * A is encoded to [1, 0 , 0, 0]
             * C is encoded to [0, 1 , 0, 0]
             * G is encoded to [0, 0 , 1, 0]
             * T is encoded to [0, 0 , 0, 1]
    """
    return global_3d_mapper[arr]


def pairwise_tensor_dot(arr1, arr2):
    # use tensor dot product to perform the pairwise distances
    s1_arr_3d = return_3d_arr(arr1)
    s2_arr_3d = return_3d_arr(arr2)
    return s1_arr_3d.shape[1] - np.tensordot(s1_arr_3d, s2_arr_3d.T, ([1, 2], [1, 0]))


def pairwise_einsum_dot(arr1, arr2):
    """
        # use einsum as a faster implementation of tensor dot product
        http://ajcr.net/Basic-guide-to-einsum/
        https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
    """
    s1_arr_3d = return_3d_arr(arr1)
    s2_arr_3d = return_3d_arr(arr2)
    return s1_arr_3d.shape[1] - np.einsum('mij,lij->ml', s1_arr_3d, s2_arr_3d)


def pairwise_base_comparison(arr1, arr2):
    """
        Rather than conver to 3d array and perform dot product, simply compare all possible NxM bases and take sum where
        N = # of sequences in arr1
        M = # of sequences in arr2
    """
    # return (return_2d_arr(arr1)[:, np.newaxis] != return_2d_arr(arr2)).sum(axis=2)
    # even faster implementation => use einsum to perform axis sum
    return np.einsum('ijk->ij', (arr1[:, np.newaxis] != arr2).astype(np.uint8))


def pairwise_scipy_cdist(arr1, arr2, convert_to_int=False):
    from scipy.spatial.distance import cdist
    s1_arr_2d = arr1
    s2_arr_2d = arr2
    dist_val = min(s1_arr_2d.shape[1], s2_arr_2d.shape[1])
    if convert_to_int:
        return (np.round(dist_val * (cdist(s1_arr_2d, s2_arr_2d, 'hamming')))).astype(np.int)
    else:
        return dist_val * (cdist(s1_arr_2d, s2_arr_2d, 'hamming'))


def seq_pwm_ascii_map_and_score(pwm_arr, seqs_arr, pwm_column_names='ACGT', use_log_before_sum=True, null_scores=1):
    """
        Calculate the pwm score for each sequence in an array

        A PWM matrix is converted into a matrix containing 256 column (one column for each ascii letter).
        Then the PWM scores at each ascii/base is copied to the PWM matrix (create a sparse matrix where only leters in the alphabet provided
        contains any information)

        Reindexing the scores is the same as above

        Args:
            pwm_arr (np.array): rows = position, columns = base pair in order of argument pwm_column_names

                ..note:: Number of columns

                    Number of columns must be equal to the string length in pwm_column_names

            seqs_arr (1D np.array): rows = sequence of interest

                ..note:: Sequence Length

                    Sequence length must be equal to the number of rows in pwm_arr

            pwm_column_names (string): Defines the base represented by each "column" in the pwm array

        Returns:
            scores(np.array 1D): score for each sequence
    """

    assert(pwm_arr.shape[1] == len(pwm_column_names))

    # convert sequences into a table such that each row is a sequence and each column is THE ASCII CODE for a specific letter in each seuqence
    # NxB 2d array where N = # of sequences B = length of sequence (Note the np.uint8)
    seq_as_ascii_arr = np.array(seqs_arr, dtype='S').view('S1').view(np.uint8).reshape(seqs_arr.shape[0], -1)

    assert(pwm_arr.shape[0] == seq_as_ascii_arr.shape[1])

    # instead of using letters to represent a pwm, make a larger matrix where you can accoutn for all "ascii" values
    pwm_using_asci = np.zeros((pwm_arr.shape[0], 256), dtype=np.float)
    pwm_using_asci.fill(null_scores)

    for p, l in enumerate(pwm_column_names):
        # now the score for a specific base at eahc position is represented by its ascii value and not letter
        pwm_using_asci[:, ord(l.upper())] = pwm_arr[:, p]

    # seqs_arr_mapped_to_pwm is still an NxB matrix except now each position  points to a specific column of interest in the PWM
    # we can select each "B position x PWM score" for each letter  in the matrix using the following
    weights_at_each_position = pwm_using_asci[np.arange(0, seq_as_ascii_arr.shape[1]), seq_as_ascii_arr]

    if use_log_before_sum:
        scores = np.exp(np.log(weights_at_each_position).sum(axis=1))
    else:
        scores = weights_at_each_position.sum(axis=1)

    return scores
