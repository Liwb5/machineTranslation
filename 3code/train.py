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

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent) #总时间
    rs = es - s        #总时间减去已经运行的时间等于还剩下的时间
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



def train(use_cuda, lr, net, epoches, train_loader, print_every, 
            batch_size, transformer):
    if use_cuda:
        net.cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                lr = lr)

    net.train()

    batch_count = 0
    print_loss = 0

    for epoch in range(epoches):
        for data in train_loader:
            batch_count += 1

            entext = data['en_index_list']
            enlen = data['en_lengths']
            zhgtruths = data['zh_index_list'] #used for training
            zhlen = data['zh_lengths']
            zhlabels = data['zh_labels_list'] #used for evaluating

            logits, predicts = net(entext, zhgtruths, enlen)

            
            loss = net.get_loss(logits, zhlabels)

            print_loss += loss.data[0]

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()            


            if (batch_count*batch_size) % print_every == 0:
                print_avg_loss = print_loss/print_every
                print_loss = 0
                print('epoch %d the loss is %.4f' % (epoch, print_avg_loss))

                evaluate(use_cuda, net, entext, zhgtruths, zhlabels, enlen, transformer)


def evaluate(use_cuda, net, entext, gtruths, zhlabels, enlen, transformer):
    if use_cuda:
        net.cuda()

    net.eval()

    logits, predicts = net(entext, gtruths, enlen, is_eval = True)

    en_origin = [0 for i in range(len(entext))]
    zh_predicts = [0 for i in range(len(entext))]
    zh_answer = [0 for i in range(len(entext))]
    #zh_gtruths = [0 for i in range(len(entext))]
    for i in range(len(entext)):
        en_origin[i] = transformer.index2text(entext[i], 'en')
        zh_answer[i] = transformer.index2text(zhlabels[i],'zh')
        zh_predicts[i] = transformer.index2text(predicts[i],'zh')
        #zh_gtruths[i] = transformer.index2text(gtruths[i],'zh')

        print('<', en_origin[i])
        print('=', zh_answer[i])
        #print('=', zh_gtruths[i])
        print('>',zh_predicts[i])

















