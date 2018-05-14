
aa_alphabet = list('ACDEFGHIKLMNPQRSTVWY')
dna_alphabet = list('ACTG')

dna_degenerate = {
    'U': list('UT'),
    'N': list('ACTG'),
    'W': list('AT'),
    'S': list('CG'),
    'M': list('AC'),
    'R': list('AG'),
    'Y': list('CT'),
    'B': list('CGT'),
    'D': list('AGT'),
    'H': list('ACT'),
    'V': list('ACG')
}

aa_degenerate = {
    'X': aa_alphabet
}

dna_extra = ['-']
aa_extra = ['-']

all_dna = dna_alphabet + list(dna_degenerate.keys()) + dna_extra
all_aa = aa_alphabet + list(aa_degenerate.keys()) + aa_extra
