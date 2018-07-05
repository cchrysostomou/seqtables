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

# dna_alphabet = list('ACTG') + sorted(list(set(sorted(degen_to_base.values())) - set('ACTG'))) + ['-.']
# aa_alphabet = list('ACDEFGHIKLMNPQRSTVWYX*Z-.')


