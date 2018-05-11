import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

import random
import numpy as np
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


def train(use_cuda, net, train_loader, valid_loader, 
          transformer, params, agent=None):
    
    start_time = time.time()
    
    if agent != None:
        pass
    
    if use_cuda:
        net.cuda()
        
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), 
                           lr = params['lr'])
    
    net.train()
    
    global_step = 0 #记录总共运行了多少batch
    print_loss = 0 #记录loss值
    
    for epoch in range(1, params['epoches']+1):
        batch_count = 0 #记录每个epoch有多少batch
        
        for data in train_loader:
            src_sent = data['fra_index_list']
            src_len = data['fra_lengths_list']
            tar_sent = data['eng_index_list']
            tar_len = data['eng_lengths_list']
            tar_sent_no_SOS = tar_sent[1:] #去掉第一个SOS_token
            
            if params['tf_ratio'] != None:
                tf_ratio = params['tf_ratio']
            else:
                tf_ratio = max(math.exp(-(global_step)/20000-0.1), 0.5)
            
            #probas --> B* tar_maxLen * tar_voc
            #predicts --> B * tar_maxLen
            probas, predicts = net(src_sent, src_len, tar_sent, tf_ratio)
            
            loss = net.getLoss(probas, tar_sent_no_SOS)
            
            print_loss += loss.data[0]
            
            optimizer.zero_grad()

            loss.backward() 

            optimizer.step() 
            
            batch_count += 1  #新的epoch下就会置零
            global_step += 1  #每个batch加1
            
            if global_step % params['print_every'] == 0:
                print_avg_loss = print_loss/print_every
                
                print_loss = 0
                
                #calculate BLEU score and valid loss
                #bleu_score, valid_loss = getBLEUandLoss(use_cuda, valid_loader, net, transformer)
                bleu_score = 0
                valid_loss = 0
                if agent != None:
                    agent.append(lossRecord, global_step, print_avg_loss)
                    agent.append(ssprobRecord, global_step, tf_ratio)
                    agent.append(scoreRecord, global_step, bleu_score)
                    agent.append(validLoss, global_step, valid_loss)
                
                print('epoch %d/%d | train_loss %.4f | valid_loss %.4f | score %.4f | ssprob %.3f | batch %d | global_step %d | %s' % (epoch, epoches, print_avg_loss, valid_loss, bleu_score, ssprob, batch_count, global_step, timeSince(start_time, batch_size*global_step/(epoches*9890000))))
                
            if global_step % save_model_every == 0:
                print('saving model ...')
                torch.save(net.state_dict(), '../models/lr{:.3f}_BS{:d}_tForce{:.3f}_BLEU{:.3f}_steps{:d}.model'\
                           .format(lr, batch_size, ssprob, print_avg_loss, 
                                   bleu_score, global_step))
        
             
            del logits, predicts
            del loss
                
                
                
                
                
                
                


