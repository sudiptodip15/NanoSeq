import tflearn
import numpy as np
from Inputfast5 import Data
import os
import Util

def getEvent(path_to_dir):
    f_list = os.listdir(path_to_dir)
    event_data_list = []
    for f in f_list:
        path_to_file = path_to_dir + f
        event_data_list.append(Data.get_data(path_to_file))
    return event_data_list


def getTensor_multi_output(event_data, window_size):


    num_features = 3
    current_mean = []
    std_cur = []
    dur_seq = []
    base = []
    kmer_size = len(event_data[0][0].model_state)
    kmer_map, inv_kmer_map = Util.getKmerMap(kmer_size)

    for event in event_data:
        for it in range(len(event)-window_size):
              temp_cur = []
              temp_std_cur = []
              temp_dur = []
              obs_point = event[it+window_size]
              temp_move = obs_point.move
              if (temp_move > 2):
                  continue
              temp_base = obs_point.model_state

              for obj in event[it:it+window_size]:
                  temp_cur.append(obj.mean)
                  temp_std_cur.append(obj.stdv)
                  temp_dur.append(obj.length)

              current_mean.append(temp_cur)
              std_cur.append(temp_std_cur)
              dur_seq.append(temp_dur)
              base.append(kmer_map[temp_base])



    current_mean = np.array(current_mean)
    std_cur = np.array(std_cur)
    dur_seq = np.array(dur_seq)
    Y = np.array(base).reshape(-1,1)
    samples, max_time_steps = current_mean.shape
    X = np.zeros((samples, max_time_steps, num_features))


    # Normalize Feature Vectors
    current_mean = (1/np.std(current_mean,axis=1).reshape(-1,1))*(current_mean - np.mean(current_mean,axis=1).reshape(-1,1))
    std_cur = (1/np.std(std_cur,axis=1).reshape(-1,1))*(std_cur - np.mean(std_cur,axis=1).reshape(-1,1))
    dur_seq = (1/np.std(dur_seq,axis=1).reshape(-1,1))*(dur_seq - np.mean(dur_seq,axis=1).reshape(-1,1))

    X[:,:,0] = current_mean
    X[:,:,1] = std_cur
    X[:,:,2] = dur_seq

    return X, Y


def getTensor(event_data):
    pass



if __name__=='__main__':

    window_size = 20
    kmer_size = 5
    num_classes = 4 ** kmer_size
    Ntrain = 5
    Nval = 1
    Ntest = 1
    
    
    event_data_train = getEvent('./NanoData_5mer/Train/')
    print(' Loading Train Data...')
    Xtr, Ytr = getTensor_multi_output([event_data_train[i] for i in range(Ntrain)], window_size)
    num_samples_train, max_time_step, num_features = Xtr.shape
    Ytr = tflearn.data_utils.to_categorical(Ytr, nb_classes=num_classes)
    
    print(' Loading Validation Data...')
    event_data_validate = getEvent('./NanoData_5mer/Validate/')
    Xval, Yval = getTensor_multi_output([event_data_validate[i] for i in range(Nval)], window_size)
    Yval = tflearn.data_utils.to_categorical(Yval, nb_classes=num_classes)    

    print(' Loading Test Data...')
    event_data_test = getEvent('./NanoData_5mer/Test/')
    Xtest, Ytest= getTensor_multi_output([event_data_test[i] for i in range(Ntest)], window_size)
    Ytest = tflearn.data_utils.to_categorical(Ytest, nb_classes=num_classes)
    
    # Define the model architecture 
    net = tflearn.input_data(shape=[None, max_time_step, num_features])
    net = tflearn.lstm(net, 128, return_seq=True, dropout=0.9)
    net = tflearn.lstm(net, 128, return_seq=False, dropout=0.9)
    net = tflearn.fully_connected(net,num_classes,activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.005,
                             loss='categorical_crossentropy')


    print(' Training Model...')
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(Xtr, Ytr, n_epoch=20, validation_set=(Xtest, Ytest), show_metric=True, batch_size=256)

    Y_hat = model.predict(Xtest)
    Y_hat = np.array(Y_hat)
    Y_hat = np.argmax(Y_hat, axis = 1).reshape(-1,1)
    num_test_samples = Ytest.shape[0]
    Ytest = np.argmax(Ytest, axis = 1).reshape(-1,1)
    acc = float((1.0/num_test_samples)*np.sum(Ytest == Y_hat))
    print("Test Accuracy = %f" %acc)

    









