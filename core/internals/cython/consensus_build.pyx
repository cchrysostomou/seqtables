# cimport cpython
import numpy as np
import pandas as pd
cimport numpy as np
cimport cython
from cython.view cimport array as cvarray
from libc.stdlib cimport malloc, free
from libc.string cimport strcmp
# from cpython.string cimport PyString_AsString
from libc.stdlib cimport atoi
# from libcpp.string cimport string

DTYPE_2 = np.int8
ctypedef np.int8_t DTYPE_2_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef bayes_prob_cons(
    np.ndarray[DTYPE_2, ndim=2]  seqs,
    np.ndarray[DTYPE_2, ndim=2]  quals,
):


    cdef Py_ssize_t nSeq = seqs.shape[0], nLets = seqs.shape[1]



    cdef int pI, passedSeq=0, insCounter=0, tmpInsPosMarker;
    cdef char **seqR = <char**>malloc(4 * sizeof(char*))
    cdef int **insPosInfo = <int**>malloc(1 * sizeof(int*))
    cdef char* tmp
    cdef int currIndArr[2]
    currIndArr[0] = 0
    currIndArr[1] = 0
    cdef seqList = [], qualList = []

    cdef regex_result *match_vec = <regex_result*>malloc(nSeq * (3 * sizeof(int) + 1 * sizeof(int*) + 1 * sizeof(char*) + 1 * sizeof(regex_result)))
    cdef int maxPosStore = 0, tmpPosStore = 0, minPosStore = pos[0], maxTotalIns = 0;
    cdef multiindex_val = [], insData = [], successfullIndexes = []
    cdef int longestSequenceLengthFound

    for ind in range(nSeq):
        match_vec[ind] = cigar_breakdown(cigars[ind].encode(), len(cigars[ind]))
        tmpPosStore = pos[ind] + len(seqs[ind]) + match_vec[ind].totalDel - match_vec[ind].totalIns - 1
        if (match_vec[ind].totalIns > maxTotalIns):
            maxTotalIns = match_vec[ind].totalIns
        if (tmpPosStore > maxPosStore):
           maxPosStore = tmpPosStore
        if (minPosStore > pos[ind]):
           minPosStore = pos[ind]

    if  (min_pos == -1): # (min_pos < 0):
        # assume its not defined by user....
        min_pos = minPosStore

    if  (max_pos == -2):  # (min_pos > max_pos):
        # assume its not defined by user ....
        max_pos = maxPosStore

    longestSequenceLengthFound = maxPosStore - minPosStore + 1

    cdef char *qEmpty = <char*>malloc(longestSequenceLengthFound * sizeof(char))  # this is the absolute
    cdef char *sEmpty = <char*>malloc(longestSequenceLengthFound * sizeof(char))

    for ind in range(longestSequenceLengthFound):
        qEmpty[ind] = null_quality
        sEmpty[ind] = edge_gap

    seqR[0] = <char*>malloc(longestSequenceLengthFound * sizeof(char))
    seqR[1] = <char*>malloc(longestSequenceLengthFound * sizeof(char))
    seqR[2] = <char*>malloc(maxTotalIns * sizeof(char))
    seqR[3] = <char*>malloc(maxTotalIns * sizeof(char))
    insPosInfo[0] = <int*>malloc(maxTotalIns * sizeof(int))

    for ind in range(nSeq):
        if (cigars[ind][0] == '*' and len(cigars[ind]) == 1):
            continue
        extract_algn_seq(
            seqs[ind].encode(),
            quals[ind].encode(),
            pos[ind],
            match_vec[ind],
            min_pos,
            max_pos,
            seqR,
            insPosInfo,
            currIndArr,
            edge_gap,
            qEmpty,
            sEmpty,
            len(seqs[ind]),
            longestSequenceLengthFound
        )

        successfullIndexes.append(indexes[ind])

        free(match_vec[ind].eventLengths)
        free(match_vec[ind].eventTypes)

        seqList.append(
            seqR[0][:currIndArr[0]]
        )

        qualList.append(
            seqR[1][:currIndArr[0]]
        )

        if (currIndArr[1] > 0):
            # add insertion information
            insCounter = 0  # starst at 0, but will be changed to 1 in else statement below

            # tmpInsPosMarker = insPosInfo[0][0]  # = > use this if we define insertions as directly TO THE RIGHT of a base
            # for pI in range(currIndArr[1]):  # = > use this if we define insertions as directly TO THE RIGHT of a base
            tmpInsPosMarker = insPosInfo[0][currIndArr[1] - 1]  # => use this if we define insertins as directly TO THE LEFT OF A BASE
            for pI in range(currIndArr[1] - 1, -1, -1):
                if (insPosInfo[0][pI] != tmpInsPosMarker):
                    # we are in a new position with respect to reference so update position
                    # insCounter = 1  # = > use this if we define insertions as directly TO THE RIGHT of a base
                    insCounter = -1
                    tmpInsPosMarker = insPosInfo[0][pI]
                else:
                    # change reference position with regard to where insertions occur at a position
                    # insCounter += 1  # = > use this if we define insertions as directly TO THE RIGHT of a base
                    insCounter -= 1
                multiindex_val.append(
                    # USE (-) VALUES BECAUSE WE WILL DEFINE INSERTIONS AS TO THE LEFT(!!!) OF THE BASE POSITION (i.e. -> A[CGG]TAA AND CGG REPRESENTS INSERTIONS AT BASE POSITION 2 (REPRESENTED BY T))
                    (indexes[ind], insPosInfo[0][pI], insCounter)
                    # IF WE WANTED TO DEFINE FROM RIGHT OR BASE POSITION THEN USE THE FOLLOWING???
                    # (indexes[ind], insPosInfo[0][pI] - 1, insCounter)
                )

                insData.append(
                    [
                        seqR[2][pI:pI + 1], seqR[3][pI:pI+1]
                    ]
                )
        passedSeq += 1

    free(seqR[0])
    free(seqR[1])
    free(seqR[2])
    free(seqR[3])
    free(insPosInfo[0])
    free(insPosInfo)
    free(match_vec)
    free(seqR)
    free(qEmpty)
    free(sEmpty)

    seq_table = np.array(seqList, dtype='S').view('S1').reshape(passedSeq, -1)
    qual_table = np.array(qualList, dtype='S').view('S1').reshape(passedSeq, -1)
    positions = np.arange(min_pos, max_pos + 1)

    return seq_table, qual_table, \
            positions, \
            multiindex_val, \
            np.array(insData, dtype='S').view('S1').reshape(len(insData), -1) if insData else np.array(insData), \
            successfullIndexes
