import TransitionMat
import ObservationMat
import DataInput
import Util
import numpy as np


class Viterbi:

    def __init__(self,Trans,Obs,num_state,t,kmer_size):
        self.Trans = Trans
        self.Obs = Obs
        self.K = num_state
        self.T = t
        self.kmer_size = kmer_size

    def decode(self,currentSeq):
        prob_mat = np.ones((self.K,self.T))*(-np.Inf)
        ptr_mat = np.zeros((self.K,self.T),dtype='int')
        kmer_map, inv_kmer_map = Util.getKmerMap(self.kmer_size)


        # Initial time step estimates
        for k in range(self.K):
           ptr_mat[k][0] = -1
           if (self.Obs[k][0] == 0 or self.Obs[k][1] == 0):
                continue

           log_emit_prob =  -(currentSeq[0]-self.Obs[k][0])**2/(2*self.Obs[k][1]**2) \
                            - np.log(self.Obs[k][1] * np.sqrt(2*np.pi))
           prob_mat[k][0] = np.log(1.0/self.K) + log_emit_prob

        print("Total Sequence Length = %d" %self.T)
        for i in range(1,self.T):
            # print(i)
            for kmer in kmer_map:
                j = kmer_map[kmer]
                if(self.Obs[j][1] == 0 or self.Obs[j][2]<10):
                    continue

                log_emit_prob = -(currentSeq[i] - self.Obs[j][0]) ** 2 / (2 * self.Obs[j][1]) \
                                - np.log(np.sqrt(2 * np.pi * self.Obs[j][1] ))
                col_list, row_list = Util.getAllowedKmerIndex(kmer, self.kmer_size, kmer_map)
                max_prob = -np.Inf
                max_k = -1
                for l in range(len(row_list)):
                    row = row_list[l]
                    col = col_list[l]
                    trans_prob = prob_mat[row][i-1] + np.log(self.Trans[row][col])
                    if(trans_prob >= max_prob):
                        max_prob = trans_prob
                        max_k = row
                prob_mat[j][i] = log_emit_prob + max_prob
                ptr_mat[j][i] = max_k

        # Traceback
        seq_est = []
        ptr = np.argmax(prob_mat[:,self.T-1])
        Y_hat = np.zeros((self.T,1))
        for i in range(self.T-1,-1,-1):
              Y_hat[i] = ptr
              kmer = inv_kmer_map[ptr]              
              seq_est.insert(0,kmer)
              ptr = ptr_mat[ptr][i]
        return Y_hat, seq_est

    def decodeConstrained(self, currentSeq, init_base):
        prob_mat = np.ones((self.K, self.T)) * (-np.Inf)
        ptr_mat = np.zeros((self.K, self.T), dtype='int')
        kmer_map, inv_kmer_map = Util.getKmerMap(self.kmer_size)        

        prob_mat[init_base][0]=0

        print("Total Sequence Length = %d" % self.T)
        for i in range(1, self.T):
            # print(i)
            for kmer in kmer_map:
                j = kmer_map[kmer]
                if (self.Obs[j][1] == 0 or self.Obs[j][2] < 10):
                    continue

                log_emit_prob = -(currentSeq[i] - self.Obs[j][0]) ** 2 / (2 * self.Obs[j][1]) \
                                - np.log(np.sqrt(2 * np.pi * self.Obs[j][1]))
                col_list, row_list = Util.getAllowedKmerIndex(kmer, self.kmer_size, kmer_map)
                max_prob = -np.Inf
                max_k = -1
                for l in range(len(row_list)):
                    row = row_list[l]
                    col = col_list[l]
                    trans_prob = prob_mat[row][i - 1] + np.log(self.Trans[row][col])
                    if (trans_prob >= max_prob):
                        max_prob = trans_prob
                        max_k = row
                prob_mat[j][i] = log_emit_prob + max_prob
                ptr_mat[j][i] = max_k

        # Traceback
        seq_est = []
        ptr = np.argmax(prob_mat[:, self.T - 1])
        Y_hat = np.zeros((self.T, 1))
        for i in range(self.T - 1, -1, -1):
            Y_hat[i] = ptr
            kmer = inv_kmer_map[ptr]            
            seq_est.insert(0, kmer)
            ptr = ptr_mat[ptr][i]
        return Y_hat, seq_est



    def decodeNaive(self,currentSeq):
        kmer_map, inv_kmer_map = Util.getKmerMap(self.kmer_size)
        seq_est = []
        Y_hat = np.zeros((self.T, 1))
        print("Total Sequence Length = %d" % self.T)
        for i in range(self.T):            
            max_prob = -np.Inf
            max_kmer = -1
            for kmer in kmer_map:
                j = kmer_map[kmer]

                if (self.Obs[j][1] == 0 or self.Obs[j][2]<5):
                    continue

                log_emit_prob = -(currentSeq[i] - self.Obs[j][0]) ** 2 / (2 * self.Obs[j][1]) \
                                - np.log( np.sqrt(2 * np.pi * self.Obs[j][1]))
                if(log_emit_prob > max_prob):
                    max_prob = log_emit_prob
                    max_kmer = kmer
            seq_est.append(max_kmer)
            Y_hat[i] = kmer_map[max_kmer]         


        return Y_hat, seq_est



if __name__=='__main__':
    kmer_size = 5
    file_list=['Event_example.txt']
    TransMat = TransitionMat.computeTransMat_event_data(kmer_size,file_list)
    ObsMat = ObservationMat.computeObsMat_event_data(kmer_size,file_list)
    num_state = 4**kmer_size

    test_file_name = 'Event_example_1.txt'
    currentSeq = DataInput.getcurrentSeq(test_file_name)
    t = len(currentSeq)
    Vit = Viterbi(TransMat,ObsMat,num_state,t,kmer_size)
    Y_hat, seq_est = Vit.decode(currentSeq)
    print(seq_est)
