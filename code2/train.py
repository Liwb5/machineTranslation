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
    if use_cuda:
        net.cuda()

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

            print('The {0} epoch of loss is'.format(epoch), loss)
            
def evaluate(use_cuda, net, eval_data, transformer):
    if use_cuda:
        net.cuda()
        
    net.eval()
    for data in eval_data:
        entext = data['en_index_list']
        enlen = data['en_lengths']
        zhgtruths = data['zh_index_list']
        zhlen = data['zh_lengths']
        zhlabels = data['zh_labels_list']

        logits, predicts = net(entext, zhgtruths, enlen)
        #对于batch中的每个句子
        en_origin = [0 for i in range(len(entext))]
        zh_predicts = [0 for i in range(len(entext))]
        zh_gtruths = [0 for i in range(len(entext))]
        for i in range(len(entext)):
            en_origin[i] = transformer.index2text(entext[i], 'en')
            zh_predicts[i] = transformer.index2text(predicts[i],'zh')
            zh_gtruths[i] = transformer.index2text(zhlabels[i],'zh')

            print('<', en_origin[i])
            print('=', zh_gtruths[i])
            print('>',zh_predicts[i])
            print(predicts[i].view(1,-1))
        break




        
        
    
    
    