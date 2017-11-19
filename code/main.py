from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np
import pandas as pd

import pickle
import jieba
import h5py

import dataProcess as dp
import models as m
import validDataProcess as vdp


SOS_token = 0
EOS_token = 1


if __name__=='__main__':
    
    #loading data
    h5py_file = h5py.File('../data/train_pairs.h5py','r')
    pairs = h5py_file['pairs']

    print(pairs[0][0].decode('utf-8'))
    print(pairs[0][1].decode('gb2312'))

    inputlang = dp.Lang('en')
    outputlang = dp.Lang('zh')
    inputlang.load('../data/inputlang.pkl')
    outputlang.load('../data/outputlang.pkl')

    print(inputlang.name,inputlang.n_words)
    print(outputlang.name,outputlang.n_words)
    #h5py_file.close()
    
    
    ##do more processing with data. cut some words
    print('cutting some words ............')
    print(len(inputlang.word2count))
    print(len(inputlang.word2index))
    print(len(inputlang.index2word))
    all_en_words = inputlang.word2count.copy()
    for word in all_en_words:
        if  all_en_words[word] <= 12:
            inputlang.word2count.pop(word)
            index = inputlang.word2index[word]
            inputlang.word2index.pop(word)
            inputlang.index2word.pop(index)

    print(len(inputlang.word2count))
    print(len(inputlang.word2index))
    print(len(inputlang.index2word))
    
    
    #train data
    hidden_size = 256
    encoder1 = m.EncoderRNN(inputlang.n_words, hidden_size)
    attn_decoder1 = m.AttnDecoderRNN(hidden_size, outputlang.n_words,
                                   1, dropout_p=0.1)

    if m.use_cuda:
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()

    m.trainIters(encoder1, attn_decoder1, inputlang, outputlang,pairs, n_iters = 8000000, \
                 plot_every = 1000, print_every=1000, save_model_every=400000,save_model_parameters=80000)
    #m.trainIters(encoder1, attn_decoder1, inputlang, outputlang,pairs, n_iters = 100, \
    #              plot_every = 1, print_every=10, save_model_every=100)    
    
    m.evaluateRandomly(encoder1,attn_decoder1,inputlang,outputlang,pairs,n = 100)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
