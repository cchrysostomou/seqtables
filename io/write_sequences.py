def seqs_to_fastq(seqdata, qualdata, output_file, header=None, append=False):
    ftype = 'a' if append else 'w'
    if header is None:
        header = ['insilica_seq_{0}'.format(i + 1) for i in range(len(seqdata))]
    with open(output_file, ftype) as w:
        for seq, quality, h in zip(list(seqdata), list(qualdata), list(header)):
            w.write('\n'.join(['@' + h, seq, '+', quality]) + '\n')

