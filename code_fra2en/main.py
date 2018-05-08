import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

import h5py
import os


from hyperboard import Agent
from Dataset import Dataset
from Transformer import Transformer
from seq2seq import Seq2Seq
from Lang import Lang
import train

params = {'epoches': 200,
                   'batch_size': 50,
                   'sentence_num': 100, #设置100表示使用100个句子作为训练集，设置None表示使用所有数据进行训练
                   'tf_ratio': None,    #测试的时候需要设置为0，训练的时候设置为None表示tf_ratio随着时间变小
                   'atten_mode': 'general',  # None 表示不使用attention，general表示使用general模式
                   
                   'lr': 0.01,
                   'dropout_p': 0,
                   'src_dims': 256,
                   'tar_dims': 256,
                   'src_hidden_state': 256,
                   'tar_hidden_state': 256,
                   'tar_maxLen': 80,
                   
                   'print_every': 2,   #每多少个batch就print一次
                   'save_model_every': 10000000  #设置多少个batch就保存一次模型
                  }

print(params)

p = {0:'SOS_token',
     1:'EOS_token',
     2:'PAD_token'
     }

os.environ["CUDA_VISIBLE_DEVICES"] = '1'#

use_cuda = torch.cuda.is_available()


if __name__ == '__main__':
    path = os.path.dirname(__file__) #获得本文件所在的目录
    if path != "":
        os.chdir(path) #将当前路径设置为本文件所在的目录，方便下面读取文件。
    
    version = input('please input the version of the trainData that you want to train')
    
    #加载数据，为了可以使用dataLoader批量加载数据，需要定义一个Dataset类，
    #按照pytorch的说明，定义好几个必要的函数后就可以使用dataLoader加载了，详情看Dataset文件。
    trainDataset = Dataset('../dataAfterProcess/dataset_fra2eng_%s.h5py'%(version),is_eval = False, num = params['sentence_num'])
    train_loader = DataLoader(trainDataset,
                         batch_size=params['batch_size'],
                         num_workers=0,   #多进程，并行加载
                         shuffle=False)

    
    validDataset =  Dataset('../dataAfterProcess/valid3.h5', is_eval = False, num = 100)
    valid_loader = DataLoader(validDataset,
                     batch_size = 50,
                     num_workers = 0,
                     shuffle = False)
    
    
    #加载两个语言库
    inputlang = Lang('fra')
    outputlang = Lang('eng')
    inputlang.load('../dataAfterProcess/fra_input_dict_%s.pkl'%(version))
    outputlang.load('../dataAfterProcess/eng_output_dict_%s.pkl'%(version))
    
    tf = Transformer(inputlang, outputlang, p)
    
    #用于画各种曲线，方便我们调试
    agent = None #Agent(address='127.0.0.1',port=5000)
    #agent = Agent(address='127.0.0.1',port=5000)
    
    
    print('%s dataset has %d words. '%(inputlang.name,inputlang.n_words))
    print('%s dataset has %d words. '%(outputlang.name,outputlang.n_words))
    
    weight = [1 for i in range(outputlang.n_words)]
    weight[2] = 0  #weight[2]对应的是padding符号，只是为了补全句子的长度，不需要计算loss
    
    net = Seq2Seq(use_cuda = use_cuda,
                  src_voc = inputlang.n_words,
                  src_dims = params['src_dims'],
                  src_hidden_size = params['src_hidden_size'],
                  tar_voc = outputlang.n_words,
                  tar_dims = params['tar_dims'],
                  tar_hidden_size = params['tar_hidden_size'],
                  dropout_p = params['dropout_p'],
                  weight = weight,
                  tar_maxLen = params['tar_maxLen'],
                  batch_size = params['batch_size'],
                  atten_mode = params['atten_mode']
                 )
    
    train.train(use_cuda = use_cuda,
                net = net,
                train_loader = train_loader,
                valid_loader = valid_loader,
                transformer = tf,
                params = params,
                agent = agent)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    