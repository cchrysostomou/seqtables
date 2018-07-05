from __future__ import absolute_import

from collections import defaultdict
import warnings
try:
    from plotly import graph_objs as go
    plotly_installed = True
except:
    plotly_installed = False
    # warnings.warn("PLOTLY not installed so interactive plots are not available. This may result in unexpected funtionality")
import math
import numpy as np

try:
    from scipy.stats import binom
    scipy_binom_installed = True
except ImportError as e:
    scipy_binom_installed = False
    # warnings.warn("Warning: cannot seem to import binom from scipy. This impacts ability to use plogo functionality. If you would like to generate plogo please install scipy")

import pandas as pd

from seqtables.core.utils.alphabets import aa_alphabet, dna_alphabet, all_aa, all_dna

amino_acid_color_properties = defaultdict(
    lambda: {"color": "#f1f2f1", "name": "Unknown"},
    {
        'R': {"color": "#1FAABD", "name": "Arginine", "short": "Ala", "charge": 1, "hydropathy": -4.5},
        'H': {"color": "#1FAABD", "name": "Histidine", "short": "His", "charge": 1, "hydropathy": -3.2},
        'K': {"color": "#1FAABD", "name": "Lysine", "short": "Lys", "charge": 1, "hydropathy": -3.9},

        'D': {"color": "#D75032", "name": "Aspartic acid", "short": "Asp", "charge": -1, "hydropathy": -3.5},
        'E': {"color": "#D75032", "name": "Glutamic acid", "short": "Glu", "charge": -1, "hydropathy": -3.5},

        'C': {"color": "#64AD59", "name": "Cysteine", "short": "Cys", "charge": 0, "hydropathy": 2.5},
        'S': {"color": "#64AD59", "name": "Serine", "short": "Ser", "charge": 0, "hydropathy": -0.8},
        'G': {"color": "#64AD59", "name": "Glycine", "short": "Cys", "charge": 0, "hydropathy": -0.4},
        'Y': {"color": "#64AD59", "name": "Tyrosine", "short": "Tyr", "charge": 0, "hydropathy": -0.8},
        'T': {"color": "#64AD59", "name": "Threonine", "short": "Thr", "charge": 0, "hydropathy": -0.7},


        'P': {"color": "#4B3E4D", "name": "Proline", "short": "Pro", "charge": 0, "hydropathy": -1.6},
        'F': {"color": "#4B3E4D", "name": "Phenylalanine", "short": "Phe", "charge": 0, "hydropathy": 2.8},
        'V': {"color": "#4B3E4D", "name": "Valine", "short": "Val", "charge": 0, "hydropathy": 4.2},
        'L': {"color": "#4B3E4D", "name": "Leucine", "short": "Leu", "charge": 0, "hydropathy": 3.8},
        'I': {"color": "#4B3E4D", "name": "Isoleucine", "short": "Ili", "charge": 0, "hydropathy": 4.5},
        'A': {"color": "#4B3E4D", "name": "Alanine", "short": "Ala", "charge": 0, "hydropathy": 1.8},

        'M': {"color": "#E57E25", "name": "Methionine", "short": "Met", "charge": 0, "hydropathy": 1.9},
        'W': {"color": "#E57E25", "name": "Tryptophan", "short": "Trp", "charge": 0, "hydropathy": -0.9},

        'N': {"color": "#92278F", "name": "Asparagine", "short": "Asn", "charge": 0, "hydropathy": -3.5},
        'Q': {"color": "#92278F", "name": "Glutamine", "short": "Pro", "charge": 0, "hydropathy": -3.5},

        'X': {"color": "#f1f2f1", "name": "Unknown", "short": "X", "charge": 0, "hydropathy": 0},
        '*': {"color": "#f1f2f1", "name": "Unknown", "short": "X", "charge": 0, "hydropathy": 0}
    }
)

dna_colors = defaultdict(
    lambda: {"color": "#f1f2f1", "name": "Unknown"},
    {
        'A': {"color": "#1FAABD", "name": "Adenine", "short": "A"},
        'T': {"color": "#D75032", "name": "Thymine", "short": "T"},
        'G': {"color": "#4B3E4D", "name": "Guanine", "short": "G"},
        'C': {"color": "#64AD59", "name": "Cytosine", "short": "C"},
        'X': {"color": "#f1f2f1", "name": "Unknown", "short": "X"},
        'N': {"color": "#f1f2f1", "name": "Unknown", "short": "X"}
    }
)


def draw_seqlogo_barplots(seq_dist, alphabet=None, label_cutoff=0.09, use_properties=True, additional_text={}, show_consensus=True, scale_by_distance=False, title='', annotation_font_size=14, yaxistitle='', bargap=None, plotwidth=None, plotheight=500, num_y_ticks=3):
    """
    Uses plotly to generate a sequence logo as a series of bar graphs. This method of sequence logo is taken from the following source:

        Repository: https://github.com/ISA-tools/SequenceLogoVis

        Paper: E. Maguire, P. Rocca-Serra, S.-A. Sansone, and M. Chen, Redesigning the sequence logo with glyph-based approaches to aid interpretation, In Proceedings of EuroVis 2014, Short Paper (2014)


    Args:
        seq_dist (Dataframe):
            Rows should be unique letters, Columns should be a specific position within the sequence,

            Values should be the values that represent the frequency OR bits of a specific letter at a specific position

        alphabet (string): AA or NT

        label_cutoff (int, default=0.09):
            Defines a cutoff for when to stop adding 'text' labels to the plot (i.e. dont add labels for low freq letters)

        use_properties (bool, default=True):
            If True and if the alphabet is AA then it will color AA by their properties

        additional_text (list of tuples):
            For each tuple, element 0 is string/label, element 1 is string of letters at each position.

            i.e. additional_text = [('text', 'AACCA'), ('MORE', 'ATTTT')]. This is meant to add extra

            layers of information for the sequence. For example may you would like to include the WT sequence at the bottom

        show_consensus (bool):
            If True will show the consensus sequence at the bottom of the graph

        scale_by_distance (bool):
            If True, then each x-coordinate of the bar graph will be equal to the column position in the dataframe.

            For example if you are showing a motif in residues 10, 50, and 90 then the xcoordinates of the bars wil be 10, 50, 90 rather than 1,2,3

        annotation_font_size (int):
            size of text font

        yaxistitle (string):
            how to label the  y axis

        bargap (float, default=None):
            bargap parameter for plots. If None, then lets plotly handle it

        plotwidth (float, default=None):
            defines the total width of the plot and individual bars

        num_y_ticks (int):
            How many ytick labels should appear in plot

    Returns:
        fig (plotly.graph_objs.Figure):
            plot object for plotting in plotly

    Examples:
        >>> from seq_tables import draw_seqlogo_barplots
        >>> import pandas as pd
        >>> from plotly.offline import iplot, init_notebook_mode
        >>> init_notebook_mode()
        >>> distribution = pd.DataFrame({1: [0.9, 0.1, 0 ,0], 2: [0.5, 0.2, 0.1, 0.2], 3: [0, 0, 0, 1], 4: [0.25, 0.25, 0.25, 0.25]}, index=['A', 'C', 'T', 'G'])
        >>> plotdata = draw_seqlogo_barplots(distribution)
        >>> iplot(plotdata)


    """
    seq_dist = seq_dist.copy()
    if plotly_installed is False:
        warnings.warn('Cannot generate seq logo plots. Please install plotly')
        return None
    data = []
    if alphabet is None:
        letters = list(seq_dist.index)
        alphabet = 'nt' if set(letters).issubset(all_dna) else 'aa'
        
    annotation = []

    if alphabet.lower() == 'nt':
        colors = {c1: dna_colors[c1]['color'] for c1 in list(dna_colors.keys()) + list(seq_dist.index)}
    elif alphabet.lower() == 'aa':
        if use_properties is True:
            colors = {c1: amino_acid_color_properties[c1]['color'] for c1 in list(amino_acid_color_properties.keys()) + list(seq_dist.index)}

    labels = seq_dist.columns
    if scale_by_distance is False:
        seq_dist = seq_dist.rename(columns={r: i + 1 for i, r in enumerate(seq_dist.columns)})
        # max_dist = seq_dist.shape[1] + 1
    else:
        start = min(seq_dist.columns)
        seq_dist = seq_dist.rename(columns={r: r - start + 1 for i, r in enumerate(seq_dist.columns)})
        # max_dist = max(seq_dist.columns) + 1

    if plotwidth is None:
        plotwidth = max(400, ((350 / 6.0) * seq_dist.shape[1]))

    cnt = 0
    for i in seq_dist.columns:
        top = 0
        l = False if cnt > 0 else True
        for name, val in seq_dist.loc[:, i].sort_values().iteritems():
            top += val
            data.append(
                go.Bar(
                    y=[val],
                    x=[i],
                    name=name,
                    marker=dict(
                        color=colors[name],
                        line=dict(
                            color='white',
                            width=1.50
                        )
                    ),
                    legendgroup=name,
                    showlegend=l
                ),
            )
            if val > label_cutoff:
                annotation.append(
                    dict(
                        x=i,
                        y=top,
                        align='center',
                        xanchor='center',
                        yanchor='top',
                        text=name,
                        font=dict(color='white', size=annotation_font_size),
                        showarrow=False
                    )
                )
        cnt += 1
    consensus_seq = seq_dist.idxmax()
    consensus_seq = dict(consensus_seq)
    starting_y = -0.1 if show_consensus else 0.0
    for pos, xc in enumerate(seq_dist.columns):
        if show_consensus:
            annotation.append(
                dict(
                    x=xc,
                    y=-0.1,
                    align='center',
                    xanchor='center',
                    yanchor='top',
                    text=consensus_seq[xc],
                    font=dict(color=colors[consensus_seq[xc]], size=annotation_font_size),
                    showarrow=False
                )
            )

        for numk, rows in enumerate(additional_text):
            # key = rows[0]
            textval = rows[1]
            if pos < len(textval):
                annotation.append(
                    dict(
                        x=xc,
                        y=-0.1 * (numk + 1) + starting_y,
                        align='center',
                        xanchor='center',
                        yanchor='top',
                        text=textval[pos],
                        font=dict(color=colors[textval[pos]], size=annotation_font_size) if textval[pos] in colors else dict(color='black', size=annotation_font_size),
                        showarrow=False
                    )
                )

    if show_consensus:
        annotation.append(
            dict(
                x=-0.1,
                y=-0.1,
                align='center',
                xanchor='right',
                yanchor='top',
                text='Consensus',
                font=dict(color='black', size=14),
                showarrow=False)
        )

    for numk, rows in enumerate(additional_text):
        annotation.append(
            dict(
                x=-0.1,
                y=-0.1 * (numk + 1) + starting_y,
                align='center',
                xanchor='right',
                yanchor='top',
                text=rows[0],
                font=dict(color='black', size=12),
                showarrow=False
            )
        )

    num_y_ticks = max(3, num_y_ticks)
    miny = 0  # math.floor(seq_dist.min().min())
    maxy = math.ceil(seq_dist.max().max())
    tick_steps = ((maxy * 1.0) - 0) / (num_y_ticks - 1)
    tmp = miny
    tick_vals = []
    while tmp <= maxy:
        tick_vals.append(round(tmp, 2))
        tmp += tick_steps

    layout = go.Layout(
        barmode='stack',
        annotations=annotation,
        yaxis=dict(showgrid=False, rangemode='nonnegative', tickvals=tick_vals, zeroline=False, showline=True, title=yaxistitle, ),
        xaxis=dict(showgrid=False, rangemode='nonnegative', side='top', showline=False, zeroline=False, ticktext=labels, tickvals=seq_dist.columns),
        # xaxis2=dict(overlaying='x', side='bottom', tickvals = seq_dist.columns, ticktext = consensus_seq),
        legend=dict(traceorder='reversed'),
        width=plotwidth,
        title=title,
        height=plotheight
        # margin={'l': 90}
    )

    if bargap is not None:
        layout['bargap'] = bargap
    fig = go.Figure(data=data, layout=layout)
    return fig, data, layout


def get_bits(distribution, N, seqtype=None, alphabet=None):
    if seqtype is None:
        assert not(alphabet is None), 'An alphabet is required if seqtype is not defined'
        alphabetN = len(alphabet)
    else:
        assert seqtype.upper() in ['NT', 'AA'], "Invalid alphabet parameter. Only allow for NT or AA"
        alphabetN = len(aa_alphabet) if seqtype.upper() == 'AA' else len(dna_alphabet)

    error_correction = (1 / np.log(2)) * ((alphabetN - 1) / (2.0 * N))

    total_height = np.log2(alphabetN) - (shannon_info(distribution, 2) + error_correction / 2)

    residue_heights = (distribution * total_height)
    residue_heights[residue_heights < 0] = 0
    return residue_heights


def shannon_info(distribution, nbit=2):
    nbit = int(nbit)
    base_change = np.log(nbit)
    # avoid the stupid error with np.log(0)
    return -1 * (distribution * np.log(distribution) / base_change).sum()


def relative_entropy(freq, seqtype, bkgrnd_freq=None):
    if bkgrnd_freq is None:
        bkgrnd_freq = unbiased_freq(seqtype, freq.index, freq.columns)
    return (freq * np.log(freq.divide(bkgrnd_freq))).sum()


def unbiased_freq(seqtype, index, columns):
    constant = 1.0 / 20 if seqtype == 'AA' else 1.0 / 4
    return pd.DataFrame(constant, index=index, columns=columns)


def get_plogo(fg_counts, seqtype, bkgrnd_freq=None, use_cdf=True, use_ln=False, alpha=0.01):
    """
    Algorithm based on following:
    O'Shea JP, Chou MF, Quader SA, Ryan JK, Church GM, & Schwartz D. (2013). pLogo: A probabilistic approach to visualizing sequence motifs. Nat Methods 10, 1211-1212
    web-service: https://plogo.uconn.edu/

    Args:
            use_cdf (boolean): Dont perform the discrete solution where we calculate the binomial at every value up until k. instead use the CDF to estimate that value.

                    This is very important when using the discrete solution takes a significant amount of time to compute the statistic because it has to loop through every possible value


    """

    def get_inf_log_odds(max_seq_count, null_freq):
        """
                Given the max seq_count and null_freq, attempt to find a "max probability" cut_off for the plogo

        """
        fudge_factor = 1.0
        while True:
            # If infinity, then K is too high for a valid statistic (answer rounds to 0, need to keep decrease K until we find a value we can handle)
            K = max_seq_count / fudge_factor
            if (K < 1):
                print(max_seq_count, fudge_factor, null_freq)
                raise "something unexpected happened for calculating upper bounds"
            fill_inf = binomial_log_odds(K, max_seq_count, null_freq)
            if (fill_inf != np.inf):
                break
            fudge_factor *= 10

        if fill_inf == np.inf:
            print(max_seq_count, fudge_factor, null_freq)
            raise "Did not expect an infinity here"

        if fudge_factor == 1.0:
            return fill_inf

        # here comes ugly code, dont really need to get an exact number

        for more_fudge in range(10, 1, -1):
            K = (max_seq_count * more_fudge) / fudge_factor
            fill_inf = binomial_log_odds(K, max_seq_count, null_freq)
            if (fill_inf != np.inf):
                break

        for even_more_fudge in range(10, 1, -1):
            K = (max_seq_count * (more_fudge + (1.0 * even_more_fudge) / 10)) / fudge_factor
            fill_inf = binomial_log_odds(K, max_seq_count, null_freq)
            if (fill_inf != np.inf):
                break

        if fill_inf == np.inf:
            print(max_seq_count, fudge_factor, null_freq, fill_inf)
            # raise "Did not expect an infinity here"

        return fill_inf

    def binomial_log_odds(k, N, freq):

        """
            Discrete solution of probability using binomial should be calculated by:
                above => Pr(k, k>=K, N, p) = sum[binomial(x, N, p) for x in range(k, N+1)] (sum all pmf from k to N)
                below => Pr(k, k<=K, N, p) = sum[binomial(x, N, p) for x in range(0, k+1)] (sum all pmf from 0 to k)

            This solution takes a very very long time if N and k are very high values (basically repeating binomial millions of time), so use CDF instead
                above => binom.sf(k-1, N, p) => where sf is same as 1-cdf; and we are doing k-1 such that we include k in the sf function rather than exclude it as expected from a 1-cdf
                below => binom.cdf(k, N, p)
        """

        assert scipy_binom_installed is True, 'Please install scipy to run plogo'
        
        if use_cdf is True:
            # faster than looping for large k/N
            above_k = binom.logsf(k - 1, N, freq)
            below_k = binom.logcdf(k, N, freq)
        else:
            above_k = np.log(sum([binom.pmf(x, N, freq) for x in range(int(k), int(N) + 1)]))
            below_k = np.log(sum([binom.pmf(x, N, freq) for x in range(0, int(k) + 1)]))

        log_odds = -1 * (above_k - below_k)
        return log_odds * np.log10(np.exp(1)) if use_ln is False else log_odds

    if bkgrnd_freq is None:
        bkgrnd_freq = unbiased_freq(seqtype, fg_counts.index, fg_counts.columns)

    # use these values to determin "max/min" possile values
    null_freq = bkgrnd_freq[bkgrnd_freq > 0].min().min() / 100000.0

    max_seq_count = fg_counts.sum().max()

    fill_inf = get_inf_log_odds(max_seq_count, null_freq)

    pos_res_heights = defaultdict(lambda: defaultdict(int))

    for c in fg_counts.columns:
        fg_col = fg_counts[c]
        N = int(fg_col.sum())
        # fg_col_lookup = defaultdict(int, dict(fg_col))
        for let in list(fg_col.index):
            p = bkgrnd_freq.loc[let, c]
            k = fg_counts.loc[let, c]
            res_height = binomial_log_odds(k, N, p)

            # if res_height == np.inf:
            #     res_height = fill_inf
            # elif res_height == -1 * np.inf:
            #     res_height = -1 * fill_inf
            pos_res_heights[c][let] = res_height

    plogo = pd.DataFrame(pos_res_heights)

    max_val = plogo.stack().apply(abs).replace(np.inf, np.nan).dropna().max()

    if (max_val > fill_inf):
        warnings.warn('Warning, the predicted fill value is lower than the observed max value in table. Estimated fill value={0}, real data max={1}'.format(str(fill_inf), str(max_val)))
        fill_inf = max_val

    plogo[plogo == np.inf] = fill_inf
    plogo[plogo == -1 * np.inf] = -1 * fill_inf

    combos = fg_counts.shape[0] * fg_counts.shape[1]
    alpha_prime = alpha / (combos)
    stat_sig = np.log10(alpha_prime / (1 - alpha_prime))
    return plogo, stat_sig, -1 * stat_sig
