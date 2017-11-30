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

import score


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



def train(use_cuda, lr, net, epoches, train_loader, print_every, save_model_every,
            batch_size, transformer, agent, hyperparameters):
    
    #to display
    hyperparameters['ID'] = 'loss'
    lossRecord = agent.register(hyperparameters,'loss',True)
    hyperparameters['ID'] = 'BLEUscore'
    scoreRecord = agent.register(hyperparameters,'BLEUscore', True)
    
    if use_cuda:
        net.cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                lr = lr)

    net.train()

    global_step = 0
    print_loss = 0
    tf_ratio = 0.5

    for epoch in range(1,epoches+1):
        batch_count = 0
        for data in train_loader:

            entext = data['en_index_list']
            enlen = data['en_lengths']
            zhgtruths = data['zh_index_list'] #used for training
            zhlen = data['zh_lengths']
            zhlabels = data['zh_labels_list'] #used for evaluating
            
            #do some thing to teacher_forcing_ratio
            

            logits, predicts = net(entext, zhgtruths, enlen, teacher_forcing_ratio=tf_ratio)

            
            loss = net.get_loss(logits, zhlabels)

            print_loss += loss.data[0]

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()            
            
            del logits, predicts
            
            batch_count += 1  #新的epoch下就会置零
            global_step += 1  #每个batch加1
            if global_step % print_every == 0:
                print_avg_loss = print_loss/print_every
                agent.append(lossRecord, global_step, print_avg_loss)
                print_loss = 0

                #calculate BLEU score
                _, zh_answer, zh_predicts = evaluate(use_cuda, net, entext, zhgtruths, zhlabels, enlen, transformer)
                bleu_score = score.BLEUscore(zh_predicts, zh_answer)
                agent.append(scoreRecord, global_step, bleu_score)
                
                print('epoch %d/%d | loss %.4f | score %.4f | batch %d' % (epoch, epoches, print_avg_loss, bleu_score, batch_count))
                
                if global_step % save_model_every == 0:
                    print('saving model ...')
                    torch.save(net.state_dict(), '../models/lr{:.3f}_BS{:d}_tForce{:.3f}_BLEU{:.3f}_steps{:d}.model'\
                           .format(lr, batch_size, tf_ratio, print_avg_loss, 
                                   bleu_score, global_step))

                    
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

        #print('<', en_origin[i])
        #print('=', zh_answer[i])
        #print('=', zh_gtruths[i])
        #print('>',zh_predicts[i])
        
    net.train()
    
    return en_origin, zh_answer, zh_predicts


def printPredictsFromDataset(use_cuda, net, data_loader, transformer, count):
    
    for data in data_loader:
        
        entext = data['en_index_list']
        enlen = data['en_lengths']
        zhgtruths = data['zh_index_list'] #used for training
        zhlen = data['zh_lengths']
        zhlabels = data['zh_labels_list'] #used for evaluating

        en_origin, zh_answer, zh_predicts = evaluate(use_cuda, net, entext, 
                                   zhgtruths, zhlabels, enlen, transformer)
        
        for i in range(len(en_origin)):
            print('<', en_origin[i])
            print('=', zh_answer[i])
            print('>',zh_predicts[i]) 
            break  #只输出一个句子
        
        count -= 1   #总共显示count个句子
        if count == 0:
            break








