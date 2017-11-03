from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

#########################################
import unicodedata
import string
import re
import jieba

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
def readTrainLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    zh_lines = open('../data/train.%s'% lang1, encoding='utf-8').read().strip().split('\n')
    zh_data_list = []
    for line in zh_lines:
        seg_line = jieba.cut(line,cut_all=False)
        dic = [seg for seg in seg_line]
        zh_data_list.append(dic)

    en_lines = open('../data/train.%s'% lang2, encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    en_data_list = [[normalizeString(s) for s in l.split('\t')] for l in en_lines]

    pairs = []
    if reverse:
        for en,zh in zip(en_data_list,zh_data_list):
            pairs.append([en[0],zh])
    else:
        for en,zh in zip(en_data_list,zh_data_list):
            pairs.append([zh, en[0]])

    return pairs
##################################################

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
