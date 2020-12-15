import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import datetime
import numpy as np
import pickle

dataset = pd.read_csv('../data/GDS5420_co-exp.csv',header=None)
np_dataset = np.array(dataset.values)

#Ids = pd.read_csv('../data/GDS5420_IDs.csv')
#np_Ids = np.array(Ids.values)

with open('../data/probe_corpus.pkl','rb') as f:
    vocabulary = pickle.load(f)

l = np_dataset.shape[0]

current_time = datetime.datetime.now()
print(current_time)
print('generating vocabulary')


#for i in range(l):
#    if np_dataset[i][0] not in vocabulary:
#        vocabulary.append(np_dataset[i][0])
#    
#    if np_dataset[i][1] not in vocabulary:
#        vocabulary.append(np_dataset[i][1])
        
vocabulary_size = len(vocabulary)
    

def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x

embedding_dims = 100
W1 = Variable(torch.randn(embedding_dims,vocabulary_size).float(),requires_grad=True)
W2 = Variable(torch.randn(vocabulary_size,embedding_dims).float(),requires_grad=True)

num_epochs = 10
learning_rate = 0.1

current_time = datetime.datetime.now()
print(current_time)
print('start_learning!')

for epo in range(num_epochs):
    loss_val = 0
    for data, target in np_dataset:
        x = Variable(get_input_layer(data)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())
        
        z1 = torch.matmul(W1,x)
        z2 = torch.matmul(W2,z1)
        
        log_softmax = F.log_softmax(z2,dim=0)
        
        loss = F.nll_loss(log_softmax.view(1,-1), y_true)
        loss_val += loss.data.item()
        loss.backward()
        
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data
        
        W1.grad.data.zero_()
        W2.grad.data.zero_()
    
    print(f'Loss at epo {epo}: {loss_val/np_dataset.shape[0]}')