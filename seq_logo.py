from collections import defaultdict
import warnings
try:
    from plotly import graph_objs as go
    plotly_installed = True
except:
    plotly_installed = False
    warnings.warn("PLOTLY not installed so interactive plots are not available. This may result in unexpected funtionality")
import math

amino_acid_color_properties = defaultdict(lambda:
    {
        "color": "#f1f2f1", "name": "Unknown"
    },
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

dna_colors = defaultdict(lambda: {"color": "#f1f2f1", "name": "Unknown"}, {
    'A': {"color": "#1FAABD", "name": "Adenine", "short": "A"},
    'T': {"color": "#D75032", "name": "Thymine", "short": "T"},
    'G': {"color": "#4B3E4D", "name": "Guanine", "short": "G"},
    'C': {"color": "#64AD59", "name": "Cytosine", "short": "C"},
    'X': {"color": "#f1f2f1", "name": "Unknown", "short": "X"},
    'N': {"color": "#f1f2f1", "name": "Unknown", "short": "X"}
})


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
    warnings.warn('Currently only frequency logos are used, will allow for entropy in future')
    seq_dist = seq_dist.copy()
    if plotly_installed is False:
        warnings.warn('Cannot generate seq logo plots. Please install plotly')
        return None
    data = []
    if alphabet is None:
        letters = list(seq_dist.index)
        alphabet = 'nt' if len(set(letters) - set(['A', 'C', 'T', 'G', 'N', 'S'])) > 0 else 'aa'

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
