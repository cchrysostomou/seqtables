"""
**Set of random util methods for working with NGS analysis scripts**
"""

import re
from Bio import SeqIO

codon_table = {
	'AAA': 'K',
	'AAC': 'N',
	'AAG': 'K',
	'AAN': 'X',
	'AAT': 'N',
	'ACA': 'T',
	'ACC': 'T',
	'ACG': 'T',
	'ACN': 'T',
	'ACT': 'T',
	'AGA': 'R',
	'AGC': 'S',
	'AGG': 'R',
	'AGN': 'X',
	'AGT': 'S',
	'ANA': 'X',
	'ANC': 'X',
	'ANG': 'X',
	'ANN': 'X',
	'ANT': 'X',
	'ATA': 'I',
	'ATC': 'I',
	'ATG': 'M',
	'ATN': 'X',
	'ATT': 'I',
	'CAA': 'Q',
	'CAC': 'H',
	'CAG': 'Q',
	'CAN': 'X',
	'CAT': 'H',
	'CCA': 'P',
	'CCC': 'P',
	'CCG': 'P',
	'CCN': 'P',
	'CCT': 'P',
	'CGA': 'R',
	'CGC': 'R',
	'CGG': 'R',
	'CGN': 'R',
	'CGT': 'R',
	'CNA': 'X',
	'CNC': 'X',
	'CNG': 'X',
	'CNN': 'X',
	'CNT': 'X',
	'CTA': 'L',
	'CTC': 'L',
	'CTG': 'L',
	'CTN': 'L',
	'CTT': 'L',
	'GAA': 'E',
	'GAC': 'D',
	'GAG': 'E',
	'GAN': 'X',
	'GAT': 'D',
	'GCA': 'A',
	'GCC': 'A',
	'GCG': 'A',
	'GCN': 'A',
	'GCT': 'A',
	'GGA': 'G',
	'GGC': 'G',
	'GGG': 'G',
	'GGN': 'G',
	'GGT': 'G',
	'GNA': 'X',
	'GNC': 'X',
	'GNG': 'X',
	'GNN': 'X',
	'GNT': 'X',
	'GTA': 'V',
	'GTC': 'V',
	'GTG': 'V',
	'GTN': 'V',
	'GTT': 'V',
	'NAA': 'X',
	'NAC': 'X',
	'NAG': 'X',
	'NAN': 'X',
	'NAT': 'X',
	'NCA': 'X',
	'NCC': 'X',
	'NCG': 'X',
	'NCN': 'X',
	'NCT': 'X',
	'NGA': 'X',
	'NGC': 'X',
	'NGG': 'X',
	'NGN': 'X',
	'NGT': 'X',
	'NNA': 'X',
	'NNC': 'X',
	'NNG': 'X',
	'NNN': 'X',
	'NNT': 'X',
	'NTA': 'X',
	'NTC': 'X',
	'NTG': 'X',
	'NTN': 'X',
	'NTT': 'X',
	'TAA': '*',
	'TAC': 'Y',
	'TAG': '*',
	'TAN': 'X',
	'TAT': 'Y',
	'TCA': 'S',
	'TCC': 'S',
	'TCG': 'S',
	'TCN': 'S',
	'TCT': 'S',
	'TGA': '*',
	'TGC': 'C',
	'TGG': 'W',
	'TGN': 'X',
	'TGT': 'C',
	'TNA': 'X',
	'TNC': 'X',
	'TNG': 'X',
	'TNN': 'X',
	'TNT': 'X',
	'TTA': 'L',
	'TTC': 'F',
	'TTG': 'L',
	'TTN': 'X',
	'TTT': 'F'
}

expanded_code = {
	'W': ['A', 'T'],
	'S': ['C', 'G'],
	'M': ['A', 'C'],
	'K': ['G', 'T'],
	'R': ['A', 'G'],
	'Y': ['C', 'T'],
	'B': ['C', 'G', 'T'],
	'D': ['A', 'G', 'T'],
	'H': ['A', 'C', 'T'],
	'V': ['A', 'C', 'G'],
	'N': ['A', 'C', 'G', 'T']
}

expanded_code_with_base = expanded_code.copy()
expanded_code_with_base.update({'A': ['A'], 'C': ['C'], 'G': ['G'], 'T': ['T']})


def seq_to_regex(seq):
	"""
	Expand a barcode sequence such that we can find it using regular expression search
	i.e. NNN = [ACTG][ACTG][ACTG]
	"""
	for degen, bases in expanded_code.items():
		seq = seq.replace(degen, '[{0}]'.format(''.join(bases)))
	return seq


def open_fasta(file):
	"""
	dump wrapper for opening fasta becuase never can remember nomenclature from biopython
	"""
	with open(file, 'r') as f:
		for rec in SeqIO.parse(f, "fasta"):
			yield (rec.id, str(rec.seq).upper())


def translate(seq, frame=0, pad=False):
	if frame:
		seq = seq[frame:]
	seq = seq.upper()
	aa_seq = ""

	if len(seq) % 3 != 0:
		if pad is True:
			seq += "N" * (3 - len(seq) % 3)
		else:
			seq = seq[:-1 * (len(seq) % 3)]
	for i in range(0, len(seq), 3):
		aa_seq += codon_table[seq[i:i + 3]]
	return aa_seq


def get_codons(seq, frame=0, pad=False):
	if frame:
		seq = seq[frame:]
	seq = seq.upper()

	if len(seq) % 3 != 0:
		if pad is True:
			seq += "N" * (3 - len(seq) % 3)
		else:
			seq = seq[:-1 * (len(seq) % 3)]
	codon_list = [seq[i:i + 3] for i in range(0, len(seq), 3)]
	return codon_list


def nt_pos_to_res_num(nt_pos, AAshift=None, use_shift=True):
	"""
	OnuI nucleotide positition in fasta file refers to AA residue 9. Based on that this will return a residue #
	for a given nucleotide position

	Parameters
	----------
	nt_pos : index/position of base starting at 1, not python 0

	Returns
	-------
		aa_pos = residue number starting at 1, not python 0
		codon_pos = position of nt within codon, start at 0
	"""
	if nt_pos < 1:
		raise Exception('Error, invalid position. please use number starting at 1')
	if use_shift and AAshift:
		nt_pos = nt_pos_shift(nt_pos, AAshift)
	nt_pos -= 1
	res_num = int(nt_pos / 3)
	codon_pos = nt_pos % 3
	return res_num + 1, nt_pos + 1, codon_pos


def nt_pos_shift(nt_pos, AAshift):
	"""
		Modify nucleotide positions in our alignment to actual positions in the real wild-type sequence.

		nt_pos : base number of wild type sequence start at 1, not python 0
		AAshift : the residue corresponding to the wild type base starting at 1
	"""
	if (nt_pos < 1):
		raise Exception('Bases provided must start at 1')
	return nt_pos + (AAshift - 1) * 3


def initialize_sequences(library_design_fasta):
	"""
	Use a FASTA File to setup all required variables before the analysis. We assume fasta file contains following headers for this project:
	5_3_fwd_primer: The nucleotide sequence from 5-3 of the fwd primer used to amplify ROI
	5_3_N_dialout: The nucleotide sequence from 5-3 of the unique dialout barcode using in the N terminus/5' downstream forward primer
	3_5_C_dialout: The nucleotide sequence from 5-3 of the unique dialout barcode using hte C terminus/upstream the reverse primer. NOTE IT WILL BE THE REVERSE COMPLEMENT BECAUSE WE NEED IT WRITTING DIRECTLY AFTER 3' OF SEQUENCE
	3_5_rev_primer: The reverse complement of the reverse primer used to amplify ROI. Downstream of 3_5_C_dialout
	amplified_seq: The region in between which represents the wildtype sequence
	WTSEQ: the full length sequence we are actually aligning to, so it should not contain any Ns in it

	Essentially, what we should be given is a string broken down by characteristic/name. This string can be put back to full length as such:
	amplified_seq_5_3 = 5_3_fwd_primer+5_3_N_dialout+ampified_seq+3_5_C_dialout+3_5_rev_primer
	"""
	parameters = {each_seq[0]: each_seq[1] for each_seq in open_fasta(library_design_fasta)}
	fwd_primer = parameters.get('5_3_fwd_primer')
	rev_primer = parameters.get('3_5_rev_primer')
	dialout_n = parameters.get('5_3_N_dialout')
	dialout_c = parameters.get('3_5_C_dialout')
	seq_of_interest = parameters.get('amplified_seq')
	actual_seq = parameters.get('WTSEQ')
	return fwd_primer, rev_primer, dialout_n, dialout_c, seq_of_interest, actual_seq


def get_read_alignment_details(actual_seq, library_seq):
	"""
	Based on the sequences provided fast file by the user (ref_info) and deconvoluted by "initialize_sequences", determine where the read sequence
	from the HTS should align to the actual sequence defined in the provided fasta file

	Parameters
	----------
	actual_seq : string defining the actual sequence (not what gets library amplified)
	library_seq : string defining the actual sequence amplified for NGS. This sequence should include the degenerate bases in regions that were used for library (i.e. AAANNSAAT)

	Returns
	-------
	expected_mutations_nt : based on the alignment between library_seq and actual_seq will return where with respect to the actual_seq the sequenced library should introduce mutations
	expected_mutations_aa : returns amino acid residues for nucleotide positions defined in 'expected_mutations_nt'
	start_of_alignment : position along actual_seq where the library_seq will start.

		.. important::
			Position will start at 0 and not 1

	wt_seq_substring : the expected amplified sequence of a wildtype sequence (i.e contains no NNS)
	"""
	# what areas of gene should be mutated
	# IMPORTANT, REMEMBER: INDEX OF 0 REFERS TONUCLEOTIDE POSITIOIN 1
	# each index refers to position within the nucelotide sequence defined by actual_seq, if it has a 1, then it will be mutated by libraries

	query = seq_to_regex(actual_seq)
	alignment = re.search(seq_to_regex(library_seq), query)

	if not(alignment):
		raise Exception("something unexpected happened, check sequences")

	# THIS WILL START THE INDEX/start of alignment AT 0 NOT 1!
	start_of_alignment = alignment.span()[0]
	end_of_alignment = alignment.span()[1]

	expected_mutations_nt = [0] * len(actual_seq)
	expected_mutations_aa = [0] * int(len(actual_seq) / 3)
	for ind, b in enumerate(library_seq):
		ind += 1  # dont start indexes at 0, start at 1
		wt_pos = ind + start_of_alignment
		if b not in ['A', 'C', 'T', 'G']:
			expected_mutations_nt[wt_pos - 1] = 1  # use python indexing
			expected_mutations_aa[nt_pos_to_res_num(wt_pos, False)[0] - 1] = 1  # use python indexing = 1
	wt_seq_substring = actual_seq[start_of_alignment:end_of_alignment]
	return expected_mutations_nt, expected_mutations_aa, start_of_alignment, wt_seq_substring
