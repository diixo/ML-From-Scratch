
from __future__ import print_function
import numpy as np
import pickle

# save numpy array as npy file
from numpy import asarray
from numpy import save

pickle_file = "../data/notMNIST.pickle"

with open(pickle_file, 'rb') as f:
   dataset = pickle.load(f, encoding='latin1')
   train_dataset  = dataset['train_dataset'] # 20k
   train_labels   = dataset['train_labels']  # 20k
   valid_dataset  = dataset['valid_dataset'] # 1k
   valid_labels   = dataset['valid_labels']  # 1k
   test_dataset   = dataset['test_dataset']  # 1k
   test_labels    = dataset['test_labels']   # 1k

   # define data
   #train_dataset.tofile('trainDataset-20k-notMNIST.bin')
   #train_labels.tofile('trainLabels-20k-notMNIST.bin')

   #valid_dataset.tofile('validDataset-1k-notMNIST.bin')
   #valid_labels.tofile('validLabels-1k-notMNIST.bin')

   #test_dataset.tofile('testDataset-1k-notMNIST.bin')
   #test_labels.tofile('testLabels-1k-notMNIST.bin')
