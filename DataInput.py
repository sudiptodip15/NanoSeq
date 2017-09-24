import Util


def getcurrentSeq(filename):
    currentSeq=[]
    with open(filename,'r') as fptr:
        for line in fptr:
            line = line.rstrip().split("\t")
            currentVal = float(line[0])
            currentSeq.append(currentVal)
    return currentSeq

def getData_event(event, kmer_map):
    currentSeq = []
    state_label = []
    for obj in event:
        currentVal = obj.mean
        stateVal = obj.model_state
        currentSeq.append(currentVal)
        state_label.append(kmer_map[stateVal])
    return currentSeq, state_label

