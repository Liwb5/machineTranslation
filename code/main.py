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
import models

SOS_token = 0
EOS_token = 1





def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)




if __name__=='__main__':
    
    #loading data
    print('loading data ...............')
    h5py_file = h5py.File('../data/train_afterProcess.h5py','r')
    pairs = h5py_file['pairs']
    print(pairs[0][0].decode('utf-8'))
    print(pairs[0][1].decode('gb2312'))

    inputlang = dp.Lang('en')
    outputlang = dp.Lang('zh')
    inputlang.load('../data/en_train.pkl')
    outputlang.load('../data/zh_train.pkl')
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
        if  all_en_words[word] <= 10:
            inputlang.word2count.pop(word)
            index = inputlang.word2index[word]
            inputlang.word2index.pop(word)
            inputlang.index2word.pop(index)

    print(len(inputlang.word2count))
    print(len(inputlang.word2index))
    print(len(inputlang.index2word))
    
    #filter some Pair for test
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
