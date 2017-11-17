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

SOS_token = 0
EOS_token = 1



class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    #通过不断输入sentence（字符串的格式），构建词与下标的对应（词典），方便制作one-hot。
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
    def save(self, path):
        with open(path,'wb') as f:
            pickle.dump([self.name,self.word2index, self.word2count, self.index2word, self.n_words],f)
    
    def load(self,path):
        with open(path,'rb') as f:
            name, self.word2index, self.word2count, self.index2word, self.n_words = pickle.load(f)
        if self.name != name:
            print('error: Name error------------------------------!')
            
    def indexesFromSentence(self, sentence):
        indexes = []
        all_lang_keys = self.word2index.keys()
        for word in sentence.split(' '):
            if word in all_lang_keys:
                indexes.append(self.word2index[word])
        indexes.append(EOS_token)
        return indexes
            
##################################################################

def readFromFile(path):
    file = open(path)
    pattern = re.compile('<seg id=".*?"> (.*?) </seg>')
    result=[]
    for line in file.readlines():
        #print(line)
        item = re.findall(pattern,line)
        if item:
            #print(item[0])
            result.append(item[0])
    print('the number of all sentences is ', len(result))
    return result

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def normalizeChinese(s):
    try:
        s.encode("gb2312")
    except UnicodeEncodeError:
        return ' '
    s = re.sub(r"[~!@#$%^&* ]+",r' ', s)
    return s


#lang1 = 'zh'  lang2 = 'en'
#默认英文到中文
def readTestLangs(path):
    print("Reading lines...")
    en_lines = readFromFile(path)
    
    # Split every line into pairs and normalize
    #去掉一些标点符号
    en_data_list = [[normalizeString(s) for s in l.split('\t')] for l in en_lines]
    pairs = []
    input_lang = Lang('en')
    for en in en_data_list:
        input_lang.addSentence(en[0])
        pairs.append(en[0].encode('utf-8'))
        
    return input_lang, pairs

if __name__=='__main__':
    test_lang, test_pairs = readTestLangs('../data/test.sgm')
    
    #save data
    test_lang.save('../data/en_test.pkl')

    h5 = h5py.File('../data/test_data.h5py','w')
    h5.create_dataset('pairs',data=test_pairs,dtype = 'S400')
    h5.close()
    
        #load data
    import dataProcess as dp
    h5py_file = h5py.File('../data/test_data.h5py','r')
    test_pairs = h5py_file['pairs']
    print(len(test_pairs))
    print(test_pairs[0][0].decode('utf-8'))
    print(test_pairs[0][1].decode('gb2312'))

    test_lang = tdp.Lang('en')
    test_lang.load('../data/en_test.pkl')


    print(test_lang.name,test_lang.n_words)
    #h5py_file.close()
    
    
    
    
    
    
    
    