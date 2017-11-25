import numpy as np 
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json
import random
import torch.nn.utils.rnn as rnn_utils

def train(use_cuda, lr, net, epoches, train_loader): 
    #use_cuda = torch.cuda.is_available()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr = lr)

    net.train()

    for epoch in range(epoches):
        for data in train_loader:
            entext = data['en_index_list']
            enlen = data['en_lengths']
            zhgtruths = data['zh_index_list']
            zhlen = data['zh_lengths']
            zhlabels = data['zh_labels_list']

            logits, predicts = net(entext, zhgtruths, enlen)

            loss = net.get_loss(logits, zhlabels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            print('loss',loss)
            
def evaluate(use_cuda, net, eval_data, inputlang, outputlang):
    net.eval()
    for data in eval_data:
        entext = data['en_index_list']
        enlen = data['en_lengths']
        zhgtruths = data['zh_index_list']
        zhlen = data['zh_lengths']
        zhlabels = data['zh_labels_list']

    logits, predicts = net(entext, zhgtruths, enlen)