from itertools import product


def getKmerMap(kmer_size):

    kmer_map = {}
    inv_kmer_map = {}
    kmer_list = [''.join(i) for i in product(['A','C','G','T'], repeat = kmer_size)]
    rows_mat = len(kmer_list)
    for i in range(rows_mat):
        kmer_map[kmer_list[i]] = i
        inv_kmer_map[i] = kmer_list[i]

    return kmer_map,inv_kmer_map

def getMoveMap():
    move_map = {'': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4, 'AA': 5, 'AC': 6, 'AG': 7, 'AT': 8, \
                'CA': 9, 'CC': 10, 'CG': 11, 'CT': 12, 'GA': 13, 'GC': 14, 'GG': 15, 'GT': 16, \
                'TA': 17, 'TC': 18, 'TG': 19, 'TT': 20}
    return move_map

def getAllowedKmerIndex(kmer,kmer_size,kmer_map):
        move_map = getMoveMap()
        index_list = []
        move_list = []
        for k in move_map:
            prev_kmer = k + kmer
            index = kmer_map[prev_kmer[0:kmer_size]]
            index_list.append(index)
            move_list.append(move_map[k])
        return move_list, index_list


if __name__=='__main__':
    kmer_size = 5
    kmer = 'CGTAC'
    kmer_map, inv_kmer_map = getKmerMap(5)
    move_list, index_list = getAllowedKmerIndex(kmer,5,kmer_map)
    for i in index_list:
        print(inv_kmer_map[i])


