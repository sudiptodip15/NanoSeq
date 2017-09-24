## Synopsis

Nanopore Sequencing is a long-read sequencing technology still in its infancy, that has the potential to resolve large duplicated segments of the genome as well as RNA transcripts 
in its entirety. The base-calling in this technology involves decoding short sequence of bases from current changes, exactly like speech processing and natural language decoding 
from morphemes. 

## Motivation

Hidden Markov Models have been used in natural language processing for a long time, until RNNs dominated the scene in the past few years. The aim of this project is to compare performance 
of these techniques for time series data amidst domain specific challenges. 

## Data

The folder `NanoData_5mer.tar.gz` only provides some sample time series files that are by no means sufficient to learn parameters of the HMM, or to train the RNN. The user can download the data 
from Oxford Nanopore and then include them in the folders for the codes to work properly. 

## Installation

- Git clone and run 'tar -xzf NanoData_5mer.tar.gz` to extract the Data folder.
- `python mainHMM.py` to obtain the decoded sequence fasta files of Test set after learning parameters of HMM viz. Transition and Observation matrix. 
- `python mainNaive.py` to learn the Observation matrix and fit a Gaussian likelihood function without incorporating time transitions.
- `python mainNanoRNN.py` to obtain the performance on test data of RNN. 

Please note that number of train, test and validate datasets need to changed in the code of RNN. They currently have some default values. 

## Conclusion

RNN/LSTM performs better on base coding than models such as HMM and Naive decoding because they are able to learn irregularities in the data, biases and unknown phenomenon not modeled in the transition state diagrams.

 
