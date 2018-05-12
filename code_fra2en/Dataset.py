import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json
import random
import h5py
from torch.utils import data

#---------------------------------
#这个文件是用于加载数据的。
#原始数据被处理后时保存在了h5py文件格式里，
#利用pytorch的torch.utils.data模块可以批量加载数据的同时不用将全部数据放到内存
#---------------------------------


class Dataset(data.Dataset):
    def __init__(self, filepath, is_eval=False, num=None):
        """
        brief: 有时候我们只是想拿一部分句子训练，看看效果，通过设置num参数就可以做到.
        """
        self.file = h5py.File(filepath, 'r')
        self.is_eval = is_eval
        self.num = num
        
        #使用多少句子进行训练
        #有时候我们只是想拿一部分句子训练，看看效果，通过设置num参数就可以做到
        if self.num != None:
            print('use %d sentences to train'% self.num)
            self.fra_index_list = self.file['fra_index_list'][0:self.num]
            self.fra_lengths_list = self.file['fra_lengths_list'][0:self.num]
            self.eng_index_list = self.file['eng_index_list'][0:self.num]
            self.eng_lengths_list = self.file['eng_lengths_list'][0:self.num]
            self.eng_label_list = self.file['eng_label_list'][0:self.num]
            
            self.nb_samples = len(self.fra_index_list)#样本数量
        
        else:
            print('use all sentences to train')
            self.fra_index_list = self.file['fra_index_list']
            self.fra_lengths_list = self.file['fra_lengths_list']
            self.eng_index_list = self.file['eng_index_list']
            self.eng_lengths_list = self.file['eng_lengths_list']
            self.eng_label_list = self.file['eng_label_list']

            self.nb_samples = len(self.fra_index_list)#样本数量
        
        
    #读取下标为index 的样本
    def __getitem__(self, index):
        
        fra_index_list = self.fra_index_list[index]
        
        fra_lengths_list = self.fra_lengths_list[index]
        
        #如果是测试集，就只有英文了。
        if self.is_eval == True:
            return {'en_index_list': fra_index_list, 'en_lengths': fra_lengths_list}
        
        eng_index_list = self.eng_index_list[index]
        eng_lengths_list = self.eng_lengths_list[index]
        eng_label_list = self.eng_label_list[index]
        
        return {'fra_index_list': fra_index_list, 'fra_lengths_list': fra_lengths_list, \
                'eng_index_list':eng_index_list, 'eng_lengths_list':eng_lengths_list,\
                'eng_label_list':eng_label_list}      
        
        
    def __len__(self):
        return self.nb_samples