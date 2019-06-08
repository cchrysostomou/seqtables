# cimport cpython
import numpy as np
# import pandas as pd
cimport numpy as np
cimport cython
from cython.view cimport array as cvarray
from libc.stdlib cimport malloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

from libc.string cimport strcmp
# from cpython.string cimport PyString_AsString
from libc.stdlib cimport atoi
# from libcpp.string cimport string

DTYPE_2 = np.int64
ctypedef np.int64_t DTYPE_2_t

DTYPE_str = np.str
ctypedef np.str DTYPE_str_t

cdef struct regex_result:
    int numHits, totalIns, totalDel, totalClip
    int *eventLengths
    char *eventTypes


cdef regex_result cigar_breakdown(char* cigarstring, int strLen):
    """
        Breaks down a cigar string into all reported events (softclipping, insertions, deletions, matches)

        Parameters
        ----------
        cigarstring: cigar string from an aligned sequence

        Returns
        -------
        ordered_mutations (list of tuples): an ordered list of all events (type, # of bases)
        mutation_summary (dict): a dict showing the number of times events were detected (i.e. total matches, total deletions)
    """
    cdef int* tempEventCounts = <int *>PyMem_Malloc(strLen * sizeof(int))
    cdef char* tempEventTypes = <char *>PyMem_Malloc(strLen * sizeof(char))
    cdef int regex_res = 0, current_str_pos = 0, matching_pos = 0, tmpStart, tmpStop, tmpNum = 0
    cdef char tmpEv
    cdef char* regexcigar = "[0-9]+[MIDNSHP=XB]"
    cdef regex_result matches
    cdef int numI = 0, numD = 0, numC = 0
    cdef char zero = "0", nine="9"

    cdef int charNum = 0, charNumStart=-1

    while charNum < strLen:
        tmpEv = cigarstring[charNum]
        if((tmpEv < zero) | (tmpEv > nine)):
            # do nothing for : padded (P), hardclip (H), AND B
            # its not a number, but a letter that defines the cigar string (i.e. M in 235M)
            tmpNum = atoi(cigarstring[charNumStart + 1:charNum])  # determine number of events using substring preceding letter
            tempEventCounts[matching_pos] = tmpNum
            charNumStart = charNum
            if tmpEv == 'I':
                numI += tmpNum
            elif tmpEv == 'D' or tmpEv == 'N':
                # treat intron/N as deletions
                numD += tmpNum
            elif tmpEv == 'S':
                numC += tmpNum
            tempEventTypes[matching_pos] = tmpEv
            matching_pos += 1
        charNum += 1

    matches.numHits = matching_pos
    matches.eventLengths = tempEventCounts
    matches.eventTypes = tempEventTypes
    matches.totalIns = numI
    matches.totalDel = numD
    matches.totalClip = numC

    return matches


cdef void extract_algn_seq(
    char* seq,
    char* qual,
    int pos,
    regex_result cigar,
    int min_pos,
    int max_pos,
    char **result,
    int **insPosInfo,
    int *currIndStore,
    char edgeGap,
    char *qualityFiller,
    char *sequenceFiller,
    int refLen,
    int longestSequenceLengthToStore
):
    cdef int refP = pos;  # position of REFERENCE
    cdef int currInd = 0;  # position in destination READ matrix
    cdef int seqP = 0; # position in the original READ
    
    cdef int ind;
    cdef char evt
    cdef int nevt
    cdef int ntmptmp, ntmp
    cdef DTYPE_str_t tmp    
    cdef int startP, finalP, adjust

    cdef char *sf = result[0]  #<char*>malloc(seqLen * sizeof(char))
    cdef char *qf = result[1]  #<char*>malloc(seqLen * sizeof(char))
    cdef char *sI = result[2]  #<char*>malloc(cigar.totalIns * sizeof(char))
    cdef char *qI = result[3]  #<char*>malloc(cigar.totalIns * sizeof(char))
    cdef int *iP = insPosInfo[0]  #<int *>malloc(cigar.totalIns * sizeof(int))
    cdef int insInd=0, tmpIns=0

    if pos > max_pos:
        # situtations where the entire read of interest/alignment STARTS AFTER the max_pos we want
        substring(sf, sequenceFiller, 0, 0, longestSequenceLengthToStore, longestSequenceLengthToStore)
        substring(qf, qualityFiller, 0, 0, longestSequenceLengthToStore, longestSequenceLengthToStore)
        currInd = max_pos - min_pos + 1
        currIndStore[0] = currInd
        currIndStore[1] = insInd
        return

    if pos > min_pos:
        # alignmennt started after min pos, so add edge gaps  (i.e. '-', or '', or '$')
        substring(sf, sequenceFiller, 0, 0, pos - min_pos, longestSequenceLengthToStore)
        substring(qf, qualityFiller, 0, 0, pos - min_pos, longestSequenceLengthToStore)
        currInd = (pos - min_pos)

    for ind in range(cigar.numHits):
        nevt = cigar.eventLengths[ind]
        evt = cigar.eventTypes[ind]

        if evt == 'M' or evt == 'X' or evt == '=':
            if (refP < min_pos):
                adjust = (refP + nevt) - min_pos
                if adjust <= 0:
                    refP += nevt
                    seqP += nevt
                    continue
                else:
                    seqP += nevt - adjust
                    refP += nevt - adjust
                    nevt = adjust
            # assert currInd >= 0, ('b', currInd, nevt)
            # assert currInd + nevt < longestSequenceLengthFound, ('a', currInd, nevt, longestSequenceLengthFound)
            # assert seqP + nevt <= refLen, (seqP, nevt, refLen)
            substring(sf, seq, seqP, currInd, nevt, longestSequenceLengthToStore)
            substring(qf, qual, seqP, currInd, nevt, longestSequenceLengthToStore)
            refP += nevt
            seqP += nevt
            currInd += nevt
        elif evt == 'I':
            if ((refP >= min_pos) & (refP<=max_pos)):
                substring(sI, seq, seqP, insInd, nevt, longestSequenceLengthToStore)
                substring(qI, qual, seqP, insInd, nevt, longestSequenceLengthToStore)
                for tmpIns in range(nevt):
                    iP[insInd + tmpIns] = refP
                insInd += nevt
            seqP += nevt
        elif evt == 'D':
            if (refP < min_pos):
                adjust = (refP + nevt) - min_pos
                if adjust <= 0:
                    refP += nevt
                    continue
                else:
                    refP += nevt - adjust
                    nevt = adjust
            add_gaps(sf, '-', currInd, nevt, longestSequenceLengthToStore)
            add_gaps(qf, '!', currInd, nevt, longestSequenceLengthToStore)
            currInd += nevt
            refP += nevt
        elif evt == 'N':
            if (refP < min_pos):
                adjust = (refP + nevt) - min_pos
                if adjust <= 0:
                    refP += nevt
                    continue
                else:
                    refP += nevt - adjust
                    nevt = adjust
            add_gaps(sf, '.', currInd, nevt, longestSequenceLengthToStore)
            add_gaps(qf, '!', currInd, nevt, longestSequenceLengthToStore)
            currInd += nevt
            refP += nevt
        elif evt == 'S':
            seqP += nevt
        elif evt == 'H':
            print('DIDNT FIX hard clipping', evt)
            pass
        else:
            assert True, ('Unexpected error/event!!...was this sequence skipped??', evt)
            raise Exception()

    if (refP < min_pos):
        # situtations where the entire read of interest/alignment ENDS before the min_pos we want
        refP = min_pos

    # print(refP, max_pos, currInd, seqLen, currInd + max_pos - refP, max_pos - refP + 1)
    if refP <= max_pos:
        # alignment ended before max position, so add edge gaps. i.e. ('-', '', '$')
        substring(sf, sequenceFiller, 0, currInd, max_pos - refP + 1, longestSequenceLengthToStore)
        substring(qf, qualityFiller, 0, currInd, max_pos - refP + 1, longestSequenceLengthToStore)
        currInd += max_pos - refP + 1
    elif refP > max_pos + 1:        
        # assert False, ('a', refP, max_pos, (refP - max_pos - 1))
        currInd -= (refP - max_pos - 1)
    # assert currInd == 4738, (pos, refP, currInd, max_pos, max_pos - min_pos + 1)
    currIndStore[0] = currInd
    currIndStore[1] = insInd


cdef void substring(char* dest, char* src, int srcP, int destP, int cnt, int maximum_dest_index):
    cdef int i
    for i in range(cnt):
        if destP + i < maximum_dest_index: # make sure we dont have a segementation fault when copying data over
            dest[destP + i] = src[srcP + i]
    return

cdef void add_gaps(char* var, char let, int p, int nLen, int maximum_dest_index):
    cdef int i
    for i in range(p, p + nLen):
        if i < maximum_dest_index:  # make sure we dont have a segementation fault when copying data over
            var[i] = let


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef df_to_algn_arr(
    np.ndarray[DTYPE_str_t, ndim=1]  seqs,
    np.ndarray[DTYPE_str_t, ndim=1]  quals,
    np.ndarray[DTYPE_2_t, ndim=1] pos,
    np.ndarray[DTYPE_str_t, ndim=1] cigars,
    np.ndarray indexes,
    int min_pos = -1,
    int max_pos = -2,
    char edge_gap = '-',
    char null_quality = '!'
):

    cdef Py_ssize_t nSeq = seqs.shape[0], ind
    cdef int pI, passedSeq=0, insCounter=0, tmpInsPosMarker;
    cdef char **seqR = <char**>PyMem_Malloc(4 * sizeof(char*))
    cdef int **insPosInfo = <int**>PyMem_Malloc(1 * sizeof(int*))
    cdef char* tmp
    cdef int currIndArr[2]
    currIndArr[0] = 0
    currIndArr[1] = 0
    cdef seqList = [], qualList = []

    cdef regex_result *match_vec = <regex_result*>PyMem_Malloc(nSeq * (3 * sizeof(int) + 1 * sizeof(int*) + 1 * sizeof(char*) + 1 * sizeof(regex_result)))
    cdef int maxPosStore = 0, tmpPosStore = 0, minPosStore = pos[0], maxTotalIns = 0;
    cdef multiindex_val = [], insData = [], successfullIndexes = []
    cdef int longestSequenceLengthToStore

    for ind in range(nSeq):
        match_vec[ind] = cigar_breakdown(cigars[ind].encode(), len(cigars[ind]))
        tmpPosStore = pos[ind] + len(seqs[ind]) + match_vec[ind].totalDel - match_vec[ind].totalIns - 1 - match_vec[ind].totalClip
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
        
    assert min_pos < max_pos, 'Error, the minimum base position must be lower than the maximum base position: ' + str(min_pos) + ',' + str(max_pos)
    
    # print(minPosStore, maxPosStore)
    longestSequenceLengthToStore = max_pos - min_pos + 1
    print('Longest Sequence Length and Positions', longestSequenceLengthToStore, min_pos, max_pos, minPosStore, maxPosStore)

    cdef char *qEmpty = <char*>PyMem_Malloc(longestSequenceLengthToStore * sizeof(char))  # this is the absolute largest a sequence can be
    cdef char *sEmpty = <char*>PyMem_Malloc(longestSequenceLengthToStore * sizeof(char))

    for ind in range(longestSequenceLengthToStore):
        qEmpty[ind] = null_quality
        sEmpty[ind] = edge_gap

    seqR[0] = <char*>PyMem_Malloc(longestSequenceLengthToStore * sizeof(char))
    seqR[1] = <char*>PyMem_Malloc(longestSequenceLengthToStore * sizeof(char))
    seqR[2] = <char*>PyMem_Malloc(maxTotalIns * sizeof(char))
    seqR[3] = <char*>PyMem_Malloc(maxTotalIns * sizeof(char))
    insPosInfo[0] = <int*>PyMem_Malloc(maxTotalIns * sizeof(int))

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
            longestSequenceLengthToStore
        )

        successfullIndexes.append(indexes[ind])

        PyMem_Free(match_vec[ind].eventLengths)
        PyMem_Free(match_vec[ind].eventTypes)

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

    PyMem_Free(seqR[0])
    PyMem_Free(seqR[1])
    PyMem_Free(seqR[2])
    PyMem_Free(seqR[3])
    PyMem_Free(insPosInfo[0])
    PyMem_Free(insPosInfo)
    PyMem_Free(match_vec)
    PyMem_Free(seqR)
    PyMem_Free(qEmpty)
    PyMem_Free(sEmpty)

    seq_table = np.array(seqList, dtype='S').view('S1').reshape(passedSeq, -1)
    qual_table = np.array(qualList, dtype='S').view('S1').reshape(passedSeq, -1)
    positions = np.arange(min_pos, max_pos + 1)

    return seq_table, qual_table, \
            positions, \
            multiindex_val, \
            np.array(insData, dtype='S').view('S1').reshape(len(insData), -1) if insData else np.array(insData), \
            successfullIndexes


cpdef test_cpy(char *s):
    cdef int i
    cdef regex_result matches = cigar_breakdown(s.encode(), len(s))
    result = []
    for i in range(matches.numHits):
        result.append((matches.eventLengths[i], matches.eventTypes[i]))
    free(matches.eventLengths)
    free(matches.eventTypes)
    return result


cpdef test_cpy_arr(np.ndarray[DTYPE_str_t, ndim=1] cigars):
    r = []
    cdef Py_ssize_t ind
    cdef bytes temp
    for ind in range(cigars.shape[0]):
        temp = cigars[ind]
        r.append(test_cpy(temp))
    return r



