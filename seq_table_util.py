import numpy as np
import warnings
try:
    from plotly import graph_objs as go
    plotly_installed = True
except:
    plotly_installed = False
    warnings.warn("PLOTLY not installed so interactive plots are not available. This may result in unexpected funtionality")
import pandas as pd
import copy

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

dna_alphabet = list('ACTG') + sorted(list(set(sorted(degen_to_base.values())) - set('ACTG'))) + ['-.']
aa_alphabet = list('ACDEFGHIKLMNPQRSTVWYX*Z-.')


def get_quality_dist(
    qual_df, bins='fastqc', percentiles=[10, 25, 50, 75, 90], exclude_null_quality=True, sample=None, plotly_sampledata_size=20,
):
    """
        Returns the distribution of quality across the given sequence, similar to FASTQC quality seq report.

        Args:
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

    # current base positions in dataframe
    col_names = set(qual_df.columns)

    if bins is 'fastqc':
        # use default bins as defined by fastqc report
        bins = [
            1, 2, 3, 4, 5, 6, 7, 8, 9,
            (10, 14), (15, 19), (20, 24), (25, 29), (30, 34), (35, 39), (40, 44), (45, 49), (50, 54), (55, 59), (60, 64),
            (65, 69), (70, 74), (80, 84), (85, 89), (90, 94), (95, 99),
            (100, 104), (105, 109), (110, 114), (115, 119), (120, 124), (125, 129), (130, 134), (135, 139), (140, 144), (145, 149), (150, 154), (155, 159), (160, 164), (165, 169), (170, 174), (175, 179), (180, 184), (185, 189), (190, 194), (195, 199),
            (200, 204), (205, 209), (210, 214), (215, 219), (220, 224), (225, 229), (230, 234), (235, 239), (240, 244), (245, 249), (250, 254), (255, 259), (260, 264), (265, 269), (270, 274), (275, 279), (280, 284), (285, 289), (290, 294), (295, 299),
            (300, qual_df.shape[1])
        ]
        bins = [x if isinstance(x, int) else (x[0], x[1]) for x in bins]
    elif bins is 'even':
        # create an equal set of 10 bins based on df shape
        binsize = int(qual_df.shape[1] / 10)
        bins = []
        for x in range(0, qual_df.shape[1], binsize):
            c1 = qual_df.columns[x]
            c2 = qual_df.columns[min(x + binsize - 1, qual_df.shape[1] - 1)]
            bins.append((c1, c2))
    else:
        # just in case its a generator (i.e. range function)
        # convert floats to ints, otherwise keep original
        bins = [int(x) if isinstance(x, float) else x for x in bins]

    binnames = OrderedDict()
    for b in bins:
        # create names for each bin
        if isinstance(b, int):
            binnames[str(b)] = (b, b)
        elif len(b) == 2:
            binnames[str(b[0]) + '-' + str(b[1])] = (b[0], b[1])

    temp = qual_df.sample(sample) if sample else qual_df

    # define the quantile percentages we will return for each quality bin
    percentiles = [round(p, 0) for p in percentiles]
    per = copy.copy(percentiles)
    # ensure that the following percentiles will ALWAYS be present
    program_required = [0, 10, 25, 50, 75, 90, 100]
    to_add_manually = set(program_required) - set(per)
    program_added_values = {f: str(int(f)) + '%' for f in to_add_manually}
    # update percentil list
    per = sorted(per + list(to_add_manually))

    # loop through each of the binnames/bin counts
    binned_data = OrderedDict()
    graphs = []  # for storing plotly graphs

    plotlychosendata = pd.DataFrame(0, index=list(binnames.keys()), columns=['min', 'max', 'mean', 'median'])

    for name, binned_cols in binnames.items():
        if isinstance(binned_cols, int):
            # not binning together multiple positions in sequence
            binned_cols = (binned_cols, binned_cols)
        # create a list of all column/base positions listed within this bin
        set_cols = set(list(range(binned_cols[0], binned_cols[1] + 1)))
        # identify columns in dataframe that intersect with columns listed above
        sel_cols = list(col_names & set_cols)
        # select qualities within bin, unwind list into a single list
        bin_qual = temp[sel_cols].values.ravel()
        if exclude_null_quality:
            quantile_res = np.percentile(bin_qual[bin_qual > 0], per)
            plotlychosendata.loc[name, 'mean'] = bin_qual[bin_qual > 0].mean()
        else:
            quantile_res = np.percentile(bin_qual, per)
            plotlychosendata.loc[name, 'mean'] = bin_qual.mean()

        storevals = []
        userchosen = {}
        for p, qnt in zip(per, quantile_res):
            if p == 0:
                plotlychosendata.loc[name, 'min'] = qnt
            if p == 100:
                plotlychosendata.loc[name, 'max'] = qnt
            if p in program_required:
                # store the values required by the program in storevals
                storevals.append(qnt)
            if p in percentiles:
                # store original quantile values desired by user in variable percentiles
                userchosen[str(int(p)) + '%'] = qnt
            if p == 50:
                # store median
                median = qnt
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

        if plotly_installed:
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
            plotdata = None
        graphs.append(plotdata)
        binned_data[name] = userchosen

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

    return graphs, pd.DataFrame(binned_data), plotlychosendata

