import os
from Inputfast5 import Data
import TransitionMat
import ObservationMat
import Viterbi
import DataInput
import Util
import numpy as np

def write_to_file(seq_est,T,kmer_size):

    fptr = open('./HMMOut/'+'Read'+str(T)+'.fasta','w')
    read = seq_est[0]
    prev_s = seq_est[0]
    move = 0
    for s in seq_est[1:]:
         for move in range(kmer_size + 1):
             sub1 = prev_s[move:]
             sub2 = s[0:kmer_size-move]
             if(sub1 == sub2):
                 break
         read += s[kmer_size - move:]
         prev_s = s

    fptr.write('>Read\n'+read)
    fptr.close()

def getEvent(path_to_dir):
    f_list = os.listdir(path_to_dir)
    event_data_list = []
    for f in f_list:
        path_to_file = path_to_dir + f
        event_data_list.append(Data.get_data(path_to_file))
    return event_data_list

def runViterbi(TransMat, ObsMat, kmer_size, num_state, event_data_test, write_fasta):


    kmer_map, inv_kmer_map = Util.getKmerMap(kmer_size)
    total_acc = 0.0
    T = 0.0
    for event in event_data_test:
        currentSeq, state_label = DataInput.getData_event(event, kmer_map)
        t = len(currentSeq)
        Vit = Viterbi.Viterbi(TransMat, ObsMat, num_state, t, kmer_size)
        Y_hat, seq_est = Vit.decode(currentSeq)
        Y_test = np.array(state_label).reshape(-1,1)
        acc = float(np.sum(Y_hat == Y_test))/t
        total_acc += float(np.sum(Y_hat == Y_test))
        T += t
        print("Accuracy = %f" %acc)
        if write_fasta == 1:
            write_to_file(seq_est,T, kmer_size)       

    total_acc/=T
    print("Total Accuracy = %f" %total_acc)

def runConstrainedViterbi(TransMat, ObsMat, kmer_size, num_state, event_data_test, write_fasta):


    kmer_map, inv_kmer_map = Util.getKmerMap(kmer_size)
    total_acc = 0.0
    T = 0.0
    for event in event_data_test:
        currentSeq, state_label = DataInput.getData_event(event, kmer_map)
        start_kmer = inv_kmer_map[state_label[0]]        
        t = len(currentSeq)
        Vit = Viterbi.Viterbi(TransMat, ObsMat, num_state, t, kmer_size)
        Y_hat, seq_est = Vit.decodeConstrained(currentSeq, state_label[0])
        Y_test = np.array(state_label).reshape(-1,1)
        acc = float(np.sum(Y_hat == Y_test))/t
        total_acc += float(np.sum(Y_hat == Y_test))
        T += t
        print("Accuracy = %f" %acc)
        if write_fasta == 1:
            write_to_file(seq_est,T, kmer_size)        

    total_acc/=T
    print("Total Accuracy = %f" %total_acc)
    

if __name__=='__main__':

    kmer_size = 5
    num_state = 4 ** kmer_size
    event_data_train = getEvent('./NanoData_5mer/Train/')
    
    print('Computing Transition Matrix ...')
    TransMat = TransitionMat.computeTransMat_event_data(kmer_size, event_data_train)    
    print('Computing Observation Matrix...')
    ObsMat = ObservationMat.computeObsMat_event_data(kmer_size, event_data_train)    
    print('Done !')

    
    print(' Loading Test Data...')
    event_data_test = getEvent('./NanoData_5mer/Test/')
    print('Viterbi Decoding ...')
    write_fasta = 1
    runConstrainedViterbi(TransMat, ObsMat,kmer_size,num_state,event_data_test, write_fasta)
    # runViterbi(TransMat, ObsMat, kmer_size, num_state, [event_data_test[i] for i in range(1)], write_fasta)
    print('Done !')

