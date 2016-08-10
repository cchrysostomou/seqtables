import numpy as np
import warnings
try:
    from plotly import graph_objs as go
    plotly_installed = True
except:
    plotly_installed = False
    warnings.warn("PLOTLY not installed so interactive plots are not available. This may result in unexpected funtionality")
import pandas as pd

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


def get_quality_dist(qual_df, bins=None, percentiles=[0.1, 0.25, 0.5, 0.75, 0.9], exclude_null_quality=True, sample=None):
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

    temp = qual_df.replace(0, np.nan) if exclude_null_quality else qual_df.copy()
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
        if plotly_installed:
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

        return (g, plotdata, median)

    # group results using the aggregation function defined above
    grouped_data = temp.groupby(get_binned_cols, axis=1).apply(lambda groups: agg_fxn(groups, percentiles))

    labels = [b for b in binnames.keys() if b in grouped_data.keys()]
    data = pd.DataFrame(grouped_data.apply(lambda x: x[0])).transpose()[labels]

    if plotly_installed is True:
        graphs = list(grouped_data.apply(lambda x: x[1]).transpose()[labels])
        # median_vals = list(grouped_data.apply(lambda x: x[2]).transpose()[labels])
        scatter_min = go.Scatter(x=data.columns, y=data.loc['min'], mode='markers', name='min', showlegend=False)
        # scatter_median = go.Scatter(x=data.columns, y=median_vals, mode='line', name='median', line=dict(shape='spline')
        scatter_mean = go.Scatter(x=data.columns, y=data.loc['mean'], mode='line', name='mean', line=dict(shape='spline'))
        graphs.extend([scatter_min, scatter_mean])
    else:
        graphs = None

    return data, graphs
