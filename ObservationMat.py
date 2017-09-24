import Util
import numpy as np

def inputObsMat(kmer_size, file_list):

    kmer_map, inv_kmer_map = Util.getKmerMap(kmer_size)
    rows_obs_mat = len(kmer_map)
    ObsMat = np.zeros((rows_obs_mat, 2), dtype='float')
    if(file_list==[]):
        model_file = 'data/Model.txt'
        with open(model_file) as fptr:
            for line in fptr:
                line = line.rstrip().split("\t")
                row_index = kmer_map[line[0]]
                ObsMat[row_index][0] = float(line[1])
                ObsMat[row_index][1] = float(line[2])


    return ObsMat
    
def computeObsMat_file(kmer_size, file_list):

    kmer_map, inv_kmer_map = Util.getKmerMap(kmer_size)
    rows_obs_mat = len(kmer_map)
    col_obs_mat = 3
    obsMat = np.zeros((rows_obs_mat, col_obs_mat), dtype='float')

    for file in file_list:
        with open(file, 'r') as fptr:

            while True:
                line = fptr.readline()
                if line == '':
                    break
                data_cols = line.rstrip().split("\t")
                current_val = float(data_cols[0])
                called_kmer = data_cols[4]
                row_index = kmer_map[called_kmer]

                obsMat[row_index][0] += current_val
                obsMat[row_index][1] += current_val**2
                obsMat[row_index][2] += 1


    obsMat[:,0] = obsMat[:,0]/obsMat[:,2]
    obsMat[:,1] = obsMat[:,1]/obsMat[:,2] - (obsMat[:,0]**2)
    return obsMat


def computeObsMat_event_data(kmer_size, event_data):
    kmer_map, inv_kmer_map = Util.getKmerMap(kmer_size)
    rows_obs_mat = len(kmer_map)
    col_obs_mat = 3
    obsMat = np.zeros((rows_obs_mat, col_obs_mat), dtype='float')
    obsMat[:,2] = np.ones(rows_obs_mat)

    for event in event_data:
        for data_row in event:
            current_val = data_row.mean
            called_kmer = data_row.model_state
            row_index = kmer_map[called_kmer]

            obsMat[row_index][0] += current_val
            obsMat[row_index][1] += current_val ** 2
            obsMat[row_index][2] += 1

    obsMat[:, 0] = obsMat[:, 0] / obsMat[:, 2]
    obsMat[:, 1] = obsMat[:, 1] / obsMat[:, 2] - (obsMat[:, 0] ** 2)
    return obsMat
