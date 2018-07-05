try:
    from Bio import SeqIO
    bio_installed = True
except:
    SeqIO = False
    bio_installed = False
