from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import pickle
import jieba


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
            
##################################################################

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

#lang1 = 'zh'  lang2 = 'en'
#默认英文到中文
def readTrainLangs(lang1, lang2, reverse=True,fenci = False):
    print("Reading lines...")

    zh_lines = open('../data/train.%s'% lang1, encoding='utf-8').read().strip().split('\n')
    zh_lines = zh_lines[0:20]  #for test
    zh_data_list = []
    if fenci:
        #jieba 分词
        for line in zh_lines:
            seg_line = jieba.cut(line,cut_all=False)
            #dic = [seg for seg in seg_line]
            dic = ' '.join(seg_line)
            zh_data_list.append(dic)
    else: #用空格按字分开
        for line in zh_lines:
            dic = ' '.join(line)
            zh_data_list.append(dic)

    en_lines = open('../data/train.%s'% lang2, encoding='utf-8').read().strip().split('\n')
    en_lines = en_lines[0:20]  #for test
    
    # Split every line into pairs and normalize
    #去掉一些标点符号
    en_data_list = [[normalizeString(s) for s in l.split('\t')] for l in en_lines]

    pairs = []
    if reverse:
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
        for en,zh in zip(en_data_list,zh_data_list):
            pairs.append([en[0],zh])
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
        for en,zh in zip(en_data_list,zh_data_list):
            pairs.append([zh, en[0]])
            
    return input_lang, output_lang, pairs
##################################################

#这部分就是对数据进行处理的函数了，上面写的函数都会在这里被调用
#最后得到三个变量input_lang，output_lang分别是源语言和目标语言的类，包含它们各自的词典。
#pairs是一个列表，列表的元素是一个二元tuple，tuple里面的内容是一句源语言字符串，一句目标语言字符串。
def prepareData(lang1, lang2, reverse=True, fenci=False):
    input_lang, output_lang, pairs = readTrainLangs(lang1, lang2, reverse, fenci)
    print("Read %s sentence pairs" % len(pairs))
    print(pairs[0])
    #pairs = filterPairs(pairs)
    #print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


inputLang, outputLang, pairs = prepareData('zh','en')