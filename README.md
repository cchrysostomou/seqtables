# seqtables

Easier methods for working with pre-aligned sequences in pandas

## Description

This class attempts to store sequences in a pandas table such that each sequence is a unique row and each column represents characters at specific positions within the aligned sequences.
The methods provided are used to slice and extract regions of interests within sequences. If look at libraries of variants, you can isolate the regions that were mutated in the library design easily
and anlayze the error rate; if looking at antibodies you can slice the CDR regions and analyze their composition, etc. 

It also lets you associate quality score information for each sequence. This allows for quality filtering and converting bases with low quality scores.

## Dependencies
This program currently only works on python2

pandas

numpy

## Optional dependencies
plotly
