import pickle
import h5py
import os
import unicodedata
import re
import random
import numpy as np
import pandas as pd
import string
import math

from Lang import Lang

def unicodeToAscii(s):
    """
    将unicode string 转换成 ASCII
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    """
    # Lowercase, trim, and remove non-letter characters
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLang(path, params, Lang1='eng', Lang2='fra', reverse=False):
    """
    读取训练数据，建立两个词典。
    params:
        reverse：如果是法语-->英语，则需要reverse
    """

    print('reading training data...')
    lines = open(path, encoding='utf-8').read().strip().split('\n')

    dataLen = len(lines)
    randomSortSeq = random.sample(range(dataLen), dataLen)

    #shuffle the data
    pairs = []
    for index in randomSortSeq:
        pairs.append([normalizeString(s) for s in lines[index].split('\t')])

    #pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    #如果是法语-->英语，则需要reverse
    if reverse:
        print('reversing language...')
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(Lang2, params.copy())
        output_lang = Lang(Lang1, params.copy())
    else:
        input_lang = Lang(Lang1, params.copy())
        output_lang = Lang(Lang2, params.copy())

    return input_lang, output_lang, pairs


def filterPairs(pairs, maxLen=1000):
    """
    如果不需要所有数据进行训练，可以在这里将pairs减少一部分.

    params:
        maxLen: 句子长度超过maxLen的话，句子会被扔掉。
    """
    result = []
    for pair in pairs:
        if len(pair[0].split(' '))<maxLen and len(pair[1].split(' '))<maxLen:
            result.append(pair)

    return result


def sentences2index(pairs, path2save, validPath):
    """
    将句子转成对应的下标，方便训练的时候可以索引到embedding。并将下标保存下来。
    params:
        path2save: 下标保存的路径
    """
    fra_index_list = []
    eng_index_list = []
    fra_lengths_list = []
    eng_lengths_list = []
    eng_label_list = []
    print('changing the sentence to index...')
    for index in range(len(pairs)):
        fra_index = []
        #将句子转成下标
        for word in pairs[index][0].split(' '):
            if word in inputLang.word2index:
                fra_index.append(inputLang.word2index[word])

        fra_index.append(EOS_token)#添加结束符
        length = len(fra_index)
        fra_lengths_list.append(length) #保存句子实际长度
        for _ in range(length, MAX_LENGTH+1):#将长度扩展到 MAX_LENGTH
            fra_index.append(PAD_token)
        fra_index_list.append(fra_index)

        eng_index = [SOS_token]#添加起始符
        #将句子转成下标
        for word in pairs[index][1].split(' '):
            if word in outputLang.word2index:
                eng_index.append(outputLang.word2index[word])

        length = len(eng_index)
        eng_lengths_list.append(length)#保存句子实际长度
        eng_labels = eng_index.copy()
        del eng_labels[0]
        eng_labels.append(EOS_token)
        for _ in range(length, MAX_LENGTH+1):
            eng_index.append(PAD_token)
            eng_labels.append(PAD_token)

        eng_index_list.append(eng_index)
        eng_label_list.append(eng_labels)

        #print(pairs[index][0], fra_index)
        #print(pairs[index][1], eng_index)
    #valid data and train data
    N = len(fra_index_list)
    N2train = math.floor(0.85*N)
    #保存文件
    print('saving data...')
    file = h5py.File(path2save, 'w')
    file.create_dataset('fra_index_list', data=fra_index_list[:N2train])
    file.create_dataset('eng_index_list', data=eng_index_list[:N2train])
    file.create_dataset('fra_lengths_list', data=fra_lengths_list[:N2train])
    file.create_dataset('eng_lengths_list', data=eng_lengths_list[:N2train])
    file.create_dataset('eng_label_list', data=eng_label_list[:N2train])
    file.close()

    print('saving valid data ...')
    file = h5py.File(validPath, 'w')
    file.create_dataset('fra_index_list', data=fra_index_list[N2train:])
    file.create_dataset('eng_index_list', data=eng_index_list[N2train:])
    file.create_dataset('fra_lengths_list', data=fra_lengths_list[N2train:])
    file.create_dataset('eng_lengths_list', data=eng_lengths_list[N2train:])
    file.create_dataset('eng_label_list', data=eng_label_list[N2train:])
    file.close()


#-----------------------------------------------------------------------------------

MAX_LENGTH = 20
#如果要修改这三个数字，需要注意同步Lang.py文件里面
SOS_token = 0
EOS_token = 1
PAD_token = 2

params = {0:'SOS_token',
          1:'EOS_token',
          2:'PAD_token'
         }

if __name__ == '__main__':
    path = os.path.dirname(__file__) #获得本文件所在的目录
    if path != "":
        os.chdir(path) #将当前路径设置为本文件所在的目录，方便下面读取文件。

    suffix = input('please input the suffix of the input and output Lang that you want to save(i.e. 3): ')

    inputLang, outputLang, pairs = readLang('../data/eng-fra.txt', params, Lang1='eng',
                                            Lang2='fra',reverse = True)

    print('read %d sentence pairs'% (len(pairs)))

    #如果不需要所有数据进行训练，可以在这里将pairs减少一部分
    pairs = filterPairs(pairs, maxLen = MAX_LENGTH)#
    print('delete sentence longer than %d words and remain %d sentence pairs.'\
          %(MAX_LENGTH, len(pairs)))

    print(random.choice(pairs))


    print('counting words ...')
    #处理数据，统计词频，单词映射下标。
    for pair in pairs:
        inputLang.addSentence(pair[0])
        outputLang.addSentence(pair[1])

    print(inputLang.name, inputLang.n_words)
    print(outputLang.name, outputLang.n_words)

    #保存词典
    inputLang.save('../dataAfterProcess/fra_dict_%s.pkl'%(suffix))
    outputLang.save('../dataAfterProcess/eng_dict_%s.pkl'%(suffix))


    sentences2index(pairs, '../dataAfterProcess/train_fra2eng_%s.h5py'%(suffix),\
    '../dataAfterProcess/valid_fra2eng_%s.h5py'%(suffix))
