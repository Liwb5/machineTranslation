import numpy as np 
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json
import random
from torch.nn import utils
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



def train(use_cuda, lr, net, epoches, train_loader, valid_loader, print_every, save_model_every,
            batch_size, transformer, hyperparameters, tf_ratio,agent=None):
    start_time = time.time()
    
    if agent != None:
        #to display in hyperboard
        hyperparameters['ID'] = 'loss'
        lossRecord = agent.register(hyperparameters,'loss',True)
        hyperparameters['ID'] = 'BLEUscore'
        scoreRecord = agent.register(hyperparameters,'BLEUscore', True)
        hyperparameters['ID'] = 'ssprob'
        ssprobRecord = agent.register(hyperparameters, 'ssprob', True)
        hyperparameters['ID'] = 'valid_loss'
        validLoss = agent.register(hyperparameters, 'valid_loss', True)
    
    if use_cuda:
        net.cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                lr = lr)

    net.train()#nn.Module中的一个成员函数，设置为训练模式，
               #如果加了dropout，BN等，那么训练和评估就不一样了，
               # 所以这个设置可以区分在训练还是在评估。

    global_step = 0  #记录总共运行了多少batch
    print_loss = 0  #记录loss值
    

    for epoch in range(1,epoches+1):
        batch_count = 0  #记录每个epoch有多少batch
        for data in train_loader:
            src_sent = data['fra_index_list']
            src_len = data['fra_lengths_list']
            tar_sent = data['eng_index_list']
            tar_len = data['eng_lengths_list']
            tar_label = data['eng_label_list']
            #上面这些变量的都是B * maxLen的tensor
            """
            print(src_sent)
            print(src_len)
            print(tar_sent)
            print(tar_len)
            print(tar_label)
            exit()
            """
            #configure the teacher_forcing_ratio
            #平常测试的时候就可以不让ssprob随时间变化
            if tf_ratio != None:
                ssprob = tf_ratio
            else:
                ssprob = max(math.exp(-(global_step)/100000-0.1), 0.5)

            #logits --> B* L * zh_voc
            #predicts -->  B * L
            logits, predicts = net(src_sent, tar_sent, src_len, teacher_forcing_ratio=ssprob)

            loss = net.get_loss(logits, tar_label)

            print_loss += loss.data[0]

            optimizer.zero_grad()

            loss.backward() 
            
            utils.clip_grad_norm(net.parameters(), 5)

            optimizer.step()            
            
            batch_count += 1  #新的epoch下就会置零
            global_step += 1  #每个batch加1

            
            if global_step % print_every == 0:
                print_avg_loss = print_loss/print_every
                
                print_loss = 0
                #calculate BLEU score and valid loss
                #bleu_score, valid_loss = getBLEUandLoss(use_cuda, valid_loader, net, transformer)
                bleu_score = 0
                valid_loss = 0
                if agent != None:
                    agent.append(lossRecord, global_step, print_avg_loss)
                    agent.append(ssprobRecord, global_step, ssprob)
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

            
def evaluate(use_cuda, net, entext, gtruths, zhlabels, enlen, transformer):
    """
    对英文句子进行翻译，并且将下标对应成单词输出。
    entext: B*maxLen
    """
    if use_cuda:
        net.cuda()

    net.eval() #nn.Module中的一个成员函数，设置为评估模式，
               #如果加了dropout，BN等，那么训练和评估就不一样了，
               # 所以这个设置可以区分在训练还是在评估。


    logits, predicts = net(entext, gtruths, enlen, is_eval = True)
    
    loss = net.get_loss(logits, zhlabels).data[0]

    en_origin = [0 for i in range(len(entext))]
    zh_predicts = [0 for i in range(len(entext))]
    zh_answer = [0 for i in range(len(entext))]
    #zh_gtruths = [0 for i in range(len(entext))]
    for i in range(len(entext)):
        en_origin[i] = transformer.index2text(entext[i], 'fra')
        zh_answer[i] = transformer.index2text(zhlabels[i],'eng')
        zh_predicts[i] = transformer.index2text(predicts[i],'eng')
        #zh_gtruths[i] = transformer.index2text(gtruths[i],'zh')

        #print('<', en_origin[i])
        #print('=', zh_answer[i])
        #print('=', zh_gtruths[i])
        #print('>',zh_predicts[i])
    del logits, predicts
    
    net.train()
    
    return en_origin, zh_answer, zh_predicts, loss




def printPredictsFromDataset(use_cuda, net, data_loader, transformer, count):
    """
    批量对data_loader中的数据进行翻译。并输出翻译结果。
    """
    for data in data_loader:
        
        entext = data['en_index_list']
        enlen = data['en_lengths']
        zhgtruths = data['zh_index_list'] #used for training
        zhlen = data['zh_lengths']
        zhlabels = data['zh_labels_list'] #used for evaluating

        en_origin, zh_answer, zh_predicts, _ = evaluate(use_cuda, net, entext, zhgtruths, zhlabels, enlen, transformer)
        
        for i in range(len(en_origin)):
            print('<', en_origin[i])
            print('=', zh_answer[i])
            print('>',zh_predicts[i]) 
            if i == 4:
                break  #只输出 i+1 个句子
        
        count -= 1   #总共显示count个batch的句子
        if count == 0:
            break

def printPredictsFromDataset2(use_cuda, net, data_loader, transformer, count):
    """
    批量对data_loader中的数据进行翻译。并输出翻译结果。
    """
    for data in data_loader:
        
        src_sent = data['fra_index_list']
        src_len = data['fra_lengths_list']
        tar_sent = data['eng_index_list']
        tar_len = data['eng_lengths_list']
        tar_label = data['eng_label_list']
        
        en_origin, zh_answer, zh_predicts, _ = evaluate(use_cuda, net, src_sent, tar_sent, tar_label, src_len, transformer)
        
        for i in range(len(en_origin)):
            print('<', en_origin[i])
            print('=', zh_answer[i])
            print('>',zh_predicts[i]) 
            if i == 4:
                break  #只输出 i+1 个句子
        
        count -= 1   #总共显示count个batch的句子
        if count == 0:
            break



def getBLEUandLoss(use_cuda, data_loader, net, transformer):
    
    bleu_score = 0
    data_loss = 0
    count = 0
    for data in data_loader:
        count += 1
        entext = data['en_index_list']
        enlen = data['en_lengths']
        zhgtruths = data['zh_index_list'] #used for training
        zhlen = data['zh_lengths']
        zhlabels = data['zh_labels_list'] #used for evaluating    
        
        _, zh_answer, zh_predicts, loss = evaluate(use_cuda, net, entext, 
                                   zhgtruths, zhlabels, enlen, transformer)
        
        bleu_score += score.BLEUscore(zh_predicts, zh_answer)
        data_loss += loss
        

    bleu_score = bleu_score / count
    data_loss = data_loss / count
    
    del entext, enlen, zhgtruths, zhlen, zhlabels
    
    return bleu_score, data_loss

