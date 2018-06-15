try:
    from Bio import SeqIO
    bio_installed = True
except:
    SeqIO = False
    bio_installed = False

from seqtables.core import seqtables
from seqtables.core.seqtables import SeqTable

from seqtables.xarray_mods.st_merge import st_merge_arrays

