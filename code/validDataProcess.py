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
def readValidLangs(lang1, lang2, reverse=True, fenci=False):
    print("Reading lines...")
    zh_lines = readFromFile('../data/valid.en-zh.%s.sgm'% lang1)
    #zh_lines = zh_lines[0:20]  #for test
    
    zh_data_list = []
    if fenci:
        #jieba 分词
        for line in zh_lines:
            seg_line = jieba.cut(line,cut_all=False)
            #dic = [seg for seg in seg_line]
            dic = ' '.join(seg_line)
            tmp = ' '
            for char in dic.split(' '):
                val = normalizeChinese(char)
                tmp += val+' '
            zh_data_list.append(tmp)
    else: #用空格按字分开
        for line in zh_lines:
            dic = ' '.join(line)
            tmp = ' '
            for char in dic.split(' '):
                val = normalizeChinese(char)##去除生僻词
                tmp += val+' '
            zh_data_list.append(tmp)
            
    en_lines = readFromFile('../data/valid.en-zh.%s.sgm'% lang2)
    #en_lines = en_lines[0:20]  #for test
    
    # Split every line into pairs and normalize
    #去掉一些标点符号
    en_data_list = [[normalizeString(s) for s in l.split('\t')] for l in en_lines]
    pairs = []
    if reverse:
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
        for en,zh in zip(en_data_list,zh_data_list):
            input_lang.addSentence(en[0])
            output_lang.addSentence(zh)
            pairs.append([en[0].encode('utf-8'),zh.encode('gb2312')])
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
        for en,zh in zip(en_data_list,zh_data_list):
            input_lang.addSentence(zh)
            output_lang.addSentence(en[0])
            pairs.append([zh.encode('gb2312'), en[0].encode('utf-8')])
            
    return input_lang, output_lang, pairs


if __name__=="__main__":
    #read data and process data
    input_valid, output_valid, pairs = readValidLangs(lang1='zh', lang2='en')
    
    #save data
    input_valid.save('../data/en_valid.pkl')
    output_valid.save('../data/zh_valid.pkl')

    h5 = h5py.File('../data/valid_data.h5py','w')
    h5.create_dataset('pairs',data=pairs,dtype = 'S400')
    h5.close()


    
    #load data
    import dataProcess as dp
    h5py_file = h5py.File('../data/valid_data.h5py','r')
    pairs = h5py_file['pairs']
    print(len(pairs))
    print(pairs[0][0].decode('utf-8'))
    print(pairs[0][1].decode('gb2312'))

    inputlang = dp.Lang('en')
    outputlang = dp.Lang('zh')
    inputlang.load('../data/en_valid.pkl')
    outputlang.load('../data/zh_valid.pkl')

    ####测试用，上面两行注释的语句在真正运行的时候要用到的##################################################################
    # pairs = pairs[0:1000]
    # for pair in pairs:
    #     inputlang.addSentence(pair[0].decode('utf-8'))
    #     outputlang.addSentence(pair[1].decode('gb2312'))

    print(inputlang.name,inputlang.n_words)
    print(outputlang.name,outputlang.n_words)
    #h5py_file.close()