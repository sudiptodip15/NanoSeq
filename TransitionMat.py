import poretools
import numpy as np
import Util


def computeTransMat_file(kmer_size,file_list):
    pseudo_count = 1
    kmer_map, inv_kmer_map = Util.getKmerMap(kmer_size)
    rows_trans_mat = len(kmer_map)
    move_map = Util.getMoveMap()
    col_trans_mat = len(move_map)
    transMat = np.zeros((rows_trans_mat, col_trans_mat),dtype = 'float') + pseudo_count

    for file in file_list:
        with open(file,'r') as fptr:
            line = fptr.readline()
            data_cols = line.rstrip().split("\t")
            prev_kmer = data_cols[4]

            while True:
                line = fptr.readline()
                if line=='':
                    break
                data_cols = line.rstrip().split("\t")
                called_kmer = data_cols[4]
                row_index = kmer_map[prev_kmer]
                prev_kmer = called_kmer
                move = int(data_cols[6])
                if (move > 2):
                    continue
                base_append = called_kmer[kmer_size-move:]
                col_index = move_map[base_append]
                transMat[row_index][col_index] += 1

    return transMat


def computeTransMat_event_data(kmer_size, event_data):
        pseudo_count = 1
        kmer_map, inv_kmer_map = Util.getKmerMap(kmer_size)
        rows_trans_mat = len(kmer_map)
        move_map = Util.getMoveMap()
        col_trans_mat = len(move_map)
        transMat = np.zeros((rows_trans_mat, col_trans_mat), dtype='float') + pseudo_count

        for event in event_data:
            prev_kmer = event[0].model_state
            for obj in event[1:]:
                    called_kmer = obj.model_state
                    row_index = kmer_map[prev_kmer]
                    prev_kmer = called_kmer
                    move = obj.move
                    if(move > 2):
                        continue
                    base_append = called_kmer[kmer_size - move:]
                    col_index = move_map[base_append]
                    transMat[row_index][col_index] += 1


        transMat = transMat*1.0/(np.sum(transMat,axis=1).reshape(-1,1))
        return transMat


if __name__=='__main__':
    file_list = ['Event_example.txt']
    kmer_size = 5
    transMat = computeTransMat_file(kmer_size, file_list)
    num_row, num_col = transMat.shape
    row_sum = np.sum(transMat, axis=1).reshape((num_row,1))
    transMat = transMat / row_sum
    np.savetxt('TransMat.txt',transMat,fmt='%.4f')