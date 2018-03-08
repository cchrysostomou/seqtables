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



DTYPE_2 = np.int64
ctypedef np.int64_t DTYPE_2_t

DTYPE_str = np.str
ctypedef np.str DTYPE_str_t


cdef extern from "regex.h" nogil:
    # import regex.h functions
    # taken from (http://romankleiner.blogspot.com/2015/06/cython-and-regular-expressions.html)
    # from doc => "Note that we only import what we need later on, and that we tell Cython to release the GIL when executing the imported functions via the magic nogil keyword."

    ctypedef struct regmatch_t:
       int rm_so
       int rm_eo
    ctypedef struct regex_t:
       pass
    int REG_EXTENDED
    int regcomp(regex_t* preg, const char* regex, int cflags)
    int regexec(const regex_t *preg, const char *string, size_t nmatch, regmatch_t pmatch[], int eflags)
    void regfree(regex_t* preg)


cdef struct regex_result:
    int numHits, totalIns, totalDel
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
    cdef int* tempEventCounts = <int *>malloc(strLen * sizeof(int))
    cdef char* tempEventTypes = <char *>malloc(strLen * sizeof(char))
    cdef regex_t regex_obj
    cdef regmatch_t regmatch_obj[1]
    cdef int regex_res = 0, current_str_pos = 0, matching_pos = 0, tmpStart, tmpStop, tmpNum = 0
    cdef char tmpEv
    cdef char* regexcigar = "[0-9]+[A-Z]"
    cdef regex_result matches
    cdef int numI = 0, numD = 0
    cdef char zero = "0", nine="9"

    cdef int charNum = 0, charNumStart=-1

    while charNum < strLen:
        tmpEv = cigarstring[charNum]
        if((tmpEv < zero) | (tmpEv > nine)):
            # its not a number, but a letter that defines the cigar string (i.e. M in 235M)
            tmpNum = atoi(cigarstring[charNumStart + 1:charNum])  # determine number of events using substring preceding letter
            tempEventCounts[matching_pos] = tmpNum
            charNumStart = charNum
            if tmpEv == 'I':
                numI += tmpNum
            elif tmpEv == 'D':
                numD += tmpNum
            tempEventTypes[matching_pos] = tmpEv
            matching_pos += 1
        charNum += 1

    # regcomp(&regex_obj, regexcigar, REG_EXTENDED)

    # regex_res = regexec(&regex_obj, cigarstring, 1, regmatch_obj, 0)

    # while regex_res == 0:
    #     tmpStart = current_str_pos + regmatch_obj[0].rm_so
    #     tmpStop = current_str_pos + regmatch_obj[0].rm_eo
    #     # tmpNum = atoi(cigarstring[tmpStart:tmpStop - 1])
    #     tmpEv = cigarstring[tmpStop-1]
    #     tempEventCounts[matching_pos] = tmpNum
    #     tempEventTypes[matching_pos] = tmpEv
    #     if tmpEv == 'I':
    #         numI += tmpNum
    #     elif tmpEv == 'D':
    #         numD += tmpNum

    #     current_str_pos += regmatch_obj[0].rm_eo
    #     matching_pos += 1

    #     regex_res = regexec(&regex_obj, cigarstring[current_str_pos:], 1, regmatch_obj, 0)

    matches.numHits = matching_pos
    matches.eventLengths = tempEventCounts
    matches.eventTypes = tempEventTypes
    matches.totalIns = numI
    matches.totalDel = numD
    # regfree(&regex_obj)

    return matches


cdef void extract_algn_seq(
    char* seq,
    char* qual,
    int pos,
    regex_result cigar,
    int min_pos,
    int max_pos,
    int strLen,
    char **result,
    int **insPosInfo,
    int *currIndStore,
    char edgeGap
):
    cdef int ind, seqP=0, refP = pos, endPos = pos + strLen + cigar.totalDel - cigar.totalIns
    cdef char evt
    cdef int nevt
    cdef int ntmptmp, ntmp
    cdef DTYPE_str_t tmp
    cdef int currInd = 0
    cdef int startP, finalP, adjust

    if max_pos < min_pos:
        max_pos = endPos

    seqLen = max_pos - min_pos + 1

    cdef char *sf = <char*>malloc(seqLen * sizeof(char))
    cdef char *qf = <char*>malloc(seqLen * sizeof(char))
    cdef char *sI = <char*>malloc(cigar.totalIns * sizeof(char))
    cdef char *qI = <char*>malloc(cigar.totalIns * sizeof(char))
    cdef int *iP = <int *>malloc(cigar.totalIns * sizeof(int))
    cdef int insInd=0, tmpIns=0

    if pos > min_pos:
        # alignmennt started after min pos, so add edge gaps  (i.e. '-', or '', or '$')
       add_gaps(sf, edgeGap, currInd, pos - min_pos)
       add_gaps(qf, "!", currInd, pos - min_pos)
       currInd += (pos - min_pos)

    for ind in range(cigar.numHits):
        nevt = cigar.eventLengths[ind]
        evt = cigar.eventTypes[ind]

        if evt == 'M':
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

            substring(sf, seq, seqP, currInd, nevt)
            substring(qf, qual, seqP, currInd, nevt)
            refP += nevt
            seqP += nevt
            currInd += nevt
        elif evt == 'I':
            if ((refP >= min_pos) & (refP<=max_pos)):
                substring(sI, seq, seqP, insInd, nevt)
                substring(qI, qual, seqP, insInd, nevt)
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
            add_gaps(sf, "-", currInd, nevt)
            add_gaps(qf, "!", currInd, nevt)
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
            add_gaps(sf, "-", currInd, nevt)
            add_gaps(qf, "!", currInd, nevt)
            currInd += nevt
            refP += nevt
        elif evt == 'S':
            seqP += nevt
        elif evt == 'H':
            print('DIDNT FIX hard clipping', evt)
            pass
        else:
            print('Unexpected error/event!!', evt)
            # raise Exception()

    # print(refP, max_pos)
    if refP <= max_pos:
        # alignmnet ended before max position, so add edge gaps. i.e. ('-', '', '$')
        add_gaps(sf, edgeGap, currInd, max_pos - refP + 1)
        add_gaps(qf, "!", currInd, max_pos - refP + 1)
        currInd += max_pos - refP + 1
    elif refP > max_pos + 1:
        currInd -= (refP - max_pos - 1)

    result[0] = sf
    result[1] = qf
    result[2] = sI
    result[3] = qI
    insPosInfo[0] = iP

    currIndStore[0] = currInd
    currIndStore[1] = insInd


cdef void substring(char* dest, char* src, int srcP, int destP, int cnt):
    cdef int i
    for i in range(cnt):
        dest[destP + i] = src[srcP + i]
    return

cdef void add_gaps(char* var, char let, int p, int nLen):
    cdef int i
    for i in range(p, p + nLen):
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
    char edge_gap = '-'
):
    cdef Py_ssize_t nSeq = seqs.shape[0], ind
    cdef int pI, passedSeq=0, insCounter=0, tmpInsPosMarker;
    cdef char **seqR = <char**>malloc(4 * sizeof(char*))
    cdef int **insPosInfo = <int**>malloc(1 * sizeof(int*))
    cdef char* tmp
    cdef int currIndArr[2]
    currIndArr[0] = 0
    currIndArr[1] = 0
    cdef seqList = [], qualList = []

    cdef regex_result *match_vec = <regex_result*>malloc(nSeq * sizeof(regex_result))
    cdef int maxPosStore = 0, tmpPosStore = 0, minPosStore = pos[0]
    cdef multiindex_val = [], insData = [], successfullIndexes = []

    for ind in range(nSeq):
        match_vec[ind] = cigar_breakdown(cigars[ind].encode(), len(cigars[ind]))
        tmpPosStore = pos[ind] + len(seqs[ind]) + match_vec[ind].totalDel - match_vec[ind].totalIns - 1
        if (tmpPosStore > maxPosStore):
            maxPosStore = tmpPosStore
        if (minPosStore > pos[ind]):
            minPosStore = pos[ind]

    if (min_pos < 0):
        # assume its not defined by user....
        min_pos = minPosStore

    if (min_pos > max_pos):
        # assume its not defined by user ....
        max_pos = maxPosStore

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
             len(seqs[ind]),
             seqR,
             insPosInfo,
             currIndArr,
             edge_gap
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

    free(match_vec)
    free(seqR)

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



