
aa_alphabet = list('ACDEFGHIKLMNPQRSTVWY')
dna_alphabet = list('ACTG')

# M = match
#  i = insertion
#  d = deletion
#  n = deletion but use '.' instead of '-'
#  s = softclipping
#  h = hardclipping
#  p = padding, Alignment with inserted sequences fully aligned is called padded alignment and is represented as  as a silent deletion from padded reference sequence. Padded area in the read and not in the reference. For this type of event, do not do anything as neither positions exist in read OR reference
#  = = sequence match
#  X = sequence mismatch
# B= move back (we wont consider this one for now)
extended_cigar_alphabet = list("MIDNSHP=XB")

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
