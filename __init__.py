try:
    from Bio import SeqIO
    bio_installed = True
except:
    SeqIO = False
    bio_installed = False

from .core import seqtables

from .xarray_mods.st_merge import st_merge_arrays

