import numpy as np 
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

class Folder(data.Dataset):
    
    def __init__(self, filepath, is_eval, is_test):
        
        self.file = h5py.File(filepath, 'r')
        self.is_test = is_test
        self.is_eval = is_eval
        
        if self.is_test == True:
            num = 20
            self.en_index_list = self.file['en_index_list'][0:num]
            self.en_lengths = self.file['en_lengths'][0:num]
            self.zh_index_list = self.file['zh_index_list'][0:num]
            self.zh_lengths = self.file['zh_lengths'][0:num]
            self.zh_labels_list = self.file['zh_labels_list'][0:num]     
            
            self.nb_samples = len(self.en_index_list)
        
        else:
            self.en_index_list = self.file['en_index_list']
            self.en_lengths = self.file['en_lengths']
            self.zh_index_list = self.file['zh_index_list']
            self.zh_lengths = self.file['zh_lengths']
            self.zh_labels_list = self.file['zh_labels_list']  
            
            self.nb_samples = len(self.en_index_list)
        
        
    def __getitem__(self, index):
        
        en_index_list = self.en_index_list[index]
        
        en_lengths = self.en_lengths[index]
        
        if self.is_eval == True:
            return {'en_index_list': en_index_list, 'en_lengths': en_lengths}
        
        zh_index_list = self.zh_index_list[index]
        zh_lengths = self.zh_lengths[index]
        zh_labels_list = self.zh_labels_list[index]
        
        return {'en_index_list': en_index_list, 'en_lengths': en_lengths, \
                'zh_index_list':zh_index_list, 'zh_lengths':zh_lengths, \
                'zh_labels_list':zh_labels_list}      
        
        
    def __len__(self):
        return self.nb_samples