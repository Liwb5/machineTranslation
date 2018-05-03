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
__PADDING__ = 2
lang1 = 'en'
lang2 = 'zh'

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS",2:"__PADDING__"}
        self.n_words = 3  # Count SOS and EOS and __PADDING__
        
    def countWordFreq(self, sentence):
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] +=1

    #通过不断输入sentence（字符串的格式），构建词与下标的对应（词典），方便制作one-hot。
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
            
    def addWord(self, word):
        if word in self.word2count:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1
            
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
    s = re.sub(r"[? % . , ！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.]+",r'', s)
    return s




if __name__=='__main__':
    path = os.path.dirname(__file__) #获得本文件所在的目录
    if path != "":
        os.chdir(path) #将当前路径设置为本文件所在的目录，方便下面读取文件。
        
    ##open data file 
    en_lines = open('../data/train.%s'% lang1).read().strip().split('\n')
    zh_lines = open('../data/train.%s'% lang2).read().strip().split('\n')
    data_length = len(zh_lines)
    random_sort_sequence = random.sample(range(data_length),data_length)

    #random_sort_sequence = random_sort_sequence[0:100]##take 100 sentences for test .
    
    
    ##---统计词频
    inputlang = Lang('en')
    outputlang = Lang('zh')
    count=0
    for index in random_sort_sequence:
        tmp = [normalizeString(s) for s in en_lines[index].split('\t')]
        for word in tmp[0].split(' '):
            inputlang.countWordFreq(word)

        zh_sentence = normalizeChinese(zh_lines[index])
        for word in zh_sentence:
            try:
                word.encode('gb2312')
                outputlang.countWordFreq(word)
            except UnicodeEncodeError:
                count+=1
                print(count,'{0} can not encode to gb2312'.format(word))

    print(len(inputlang.word2count),len(outputlang.word2count))
    print(inputlang.n_words, outputlang.n_words)
    
    
    #删除词频比较低的词
    print(len(inputlang.word2count), inputlang.n_words)
    all_en_words = inputlang.word2count.copy()
    for word in all_en_words:
        if all_en_words[word] <= 15:
            inputlang.word2count.pop(word)
    print(len(inputlang.word2count), inputlang.n_words)
    
    #删除词频比较低的中文字
    print(len(outputlang.word2count), outputlang.n_words)
    all_zh_words = outputlang.word2count.copy()
    for word in all_zh_words:
        if all_zh_words[word] <= 10:
            outputlang.word2count.pop(word)
    print(len(outputlang.word2count), outputlang.n_words)
    
    #统计Word2index 与 index2word
    for index in random_sort_sequence:
        tmp = [normalizeString(s) for s in en_lines[index].split('\t')]
        for word in tmp[0].split(' '):
            inputlang.addWord(word)

        zh_sentence = normalizeChinese(zh_lines[index])
        for word in zh_sentence:
            try:
                word.encode('gb2312')
                outputlang.addWord(word)
            except UnicodeEncodeError:
                #print('{0} can not encode to gb2312'.format(word))
                pass

    print(len(inputlang.word2count),len(outputlang.word2count))
    print(len(inputlang.word2index),len(outputlang.word2index))
    print(len(inputlang.index2word),len(outputlang.index2word))
    print(inputlang.n_words, outputlang.n_words)

    #保存词典
    inputlang.save('../data/en_input_dict.pkl')
    outputlang.save('../data/zh_output_dict.pkl')
    
    #
    #将句子转成下标.
    en_MAX_LENGTH = 50
    zh_MAX_LENGTH = 80
    en_index_list = []
    en_lengths = []
    zh_index_list = []
    zh_lengths = []
    zh_labels_list = []

    for index in random_sort_sequence:
        en_sentence = [normalizeString(s) for s in en_lines[index].split('\t')]
        en_indexes = []
        zh_sentence = normalizeChinese(zh_lines[index])
        zh_indexes = [SOS_token]
        for word in en_sentence[0].split(' '): #将英文单词转成index
            if word in inputlang.word2index:
                en_indexes.append(inputlang.word2index[word])
        en_indexes.append(EOS_token)

        for word in zh_sentence:#讲中文字转成index
            if word in outputlang.word2index:
                zh_indexes.append(outputlang.word2index[word])

        #长度是否超过最大长度
        if len(en_indexes) > en_MAX_LENGTH or len(zh_indexes) > zh_MAX_LENGTH:
            continue
        else:
            en_lengths.append(len(en_indexes))
            zh_lengths.append(len(zh_indexes))

            #将长度扩展到最大长度
            for _ in range(en_MAX_LENGTH-len(en_indexes)):
                en_indexes.append(__PADDING__)
            for _ in range(zh_MAX_LENGTH-len(zh_indexes)):
                zh_indexes.append(__PADDING__)
            en_index_list.append(en_indexes) 
            zh_index_list.append(zh_indexes)
            # just to save a label for calculate loss 
            tmp = zh_indexes.copy()
            del tmp[0]
            tmp.append(__PADDING__)
            zh_labels_list.append(tmp)
    
    #保存文件
    h5pyFile = h5py.File('../data/train.h5','w')
    h5pyFile.create_dataset('en_index_list',data=en_index_list)
    h5pyFile.create_dataset('zh_index_list',data=zh_index_list)
    h5pyFile.create_dataset('en_lengths',data=en_lengths)
    h5pyFile.create_dataset('zh_lengths',data=zh_lengths)
    h5pyFile.create_dataset('zh_labels_list',data=zh_labels_list)
    h5pyFile.close()
    
    
    #加载文件
    file = h5py.File('../data/train.h5','r')
    en_index_list = file['en_index_list']
    zh_index_list = file['zh_index_list']
    en_lengths = file['en_lengths']
    zh_lengths = file['zh_lengths']
    zh_labels_list = file['zh_labels_list']
    
    inputlang = Lang('en')
    outputlang = Lang('zh')
    inputlang.load('../data/en_input_dict.pkl')
    outputlang.load('../data/zh_output_dict.pkl')