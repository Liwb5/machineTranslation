
import torch
import h5py
import os
import sys
import math

from Dataset import Dataset
from torch.utils.data import DataLoader
import dataProcess as dp
import Transformer2
import train
from decoder import Decoder
from encoder import Encoder
from decoder import Decoder
import enc2dec
from hyperboard import Agent


p = {0:'SOS_token',
     1:'EOS_token',
     2:'PAD_token'
     }

#os.chdir('/home/liwb/Documents/projects/mt/machineTranslation/')#修改当前路径到工程路径

os.environ["CUDA_VISIBLE_DEVICES"] = '0'#

use_cuda = torch.cuda.is_available()


sentence_num = None  #设置数字表示使用部分数据用于测试代码是否正确，设置None表示使用所有数据进行训练

atten_mode = 'general'  #None 表示不使用attention，general表示使用general模式
tf_ratio = None   #测试的时候是1，如果为None表示tf_ratio随着时间变小

batch_size = 200
en_dims = 256
zh_dims = 256
en_hidden_size = 256
zh_hidden_size = 256
zh_maxLength = 21
lr = 0.01
Epoches = 200
dropout_p = 0.1
num_layers = 2
bidirectional = True

print_every = 10 #每多少个batch就print一次
save_model_every = 2000#设置多少个batch就保存一次模型

hyperparameters = {'epoches': 200,
                   'batch_size': 50,
                   'sentence_num': 100, #设置100表示使用100个句子作为训练集，设置None表示使用所有数据进行训练
                   'tf_ratio': None,    #测试的时候需要设置为0，训练的时候设置为None表示tf_ratio随着时间变小
                   'atten_mode': 'general',  # None 表示不使用attention，general表示使用general模式
                   
                   'lr': 0.01,
                   'dropout_p': 0.1,
                   'en_dims': 256,
                   'zh_dims': 256,
                   'en_hidden_state': 256,
                   'zh_hidden_state': 256,
                   'zh_maxLength': 80,
                   
                   'print_every': 2,   #每多少个batch就print一次
                   'save_model_every': 10000000  #设置多少个batch就保存一次模型
                  }


if __name__ == '__main__':
    path = os.path.dirname(__file__) #获得本文件所在的目录
    if path != "":
        os.chdir(path) #将当前路径设置为本文件所在的目录，方便下面读取文件。
    
    version = '3'
    #加载数据，为了可以使用dataLoader批量加载数据，需要定义一个Dataset类，
    #按照pytorch的说明，定义好几个必要的函数后就可以使用dataLoader加载了，详情看Dataset文件。
    trainDataset = Dataset('../dataAfterProcess/train_fra2eng_%s.h5py'%(version),is_eval = False, num = sentence_num)
    
    train_loader = DataLoader(trainDataset,
                         batch_size=batch_size,
                         num_workers=0,#多进程，并行加载
                         shuffle=False)

    validDataset =  Dataset('../dataAfterProcess/valid_fra2eng_%s.h5py'%(version), is_eval = False, num = 100)
    valid_loader = DataLoader(validDataset,
                     batch_size = 50,
                     num_workers = 0,
                     shuffle = False)
    
    #加载两个语言库
    inputlang = dp.Lang('fra')
    outputlang = dp.Lang('eng')
    inputlang.load('../dataAfterProcess/fra_dict_%s.pkl'%(version))
    outputlang.load('../dataAfterProcess/eng_dict_%s.pkl'%(version))

    #transformer可以将词的下标转成对应的单词，方便我们查看
    tf = Transformer2.Transformer(inputlang, outputlang, p)
    
    #用于画各种曲线，方便我们调试
    agent = None #Agent(address='127.0.0.1',port=5000)
    #agent = Agent(address='127.0.0.1',port=1357)
    
    
    print('%s dataset has %d words. '%(inputlang.name,inputlang.n_words))
    print('%s dataset has %d words. '%(outputlang.name,outputlang.n_words))
    
    weight = [1 for i in range(outputlang.n_words)]
    weight[2] = 0  #weight[2]对应的是padding符号，只是为了补全句子的长度，不需要计算loss
            
    net = enc2dec.Net(use_cuda = use_cuda,
                 en_voc = inputlang.n_words,
                 en_dims = en_dims,
                 zh_voc = outputlang.n_words,
                 zh_dims = zh_dims,
                 en_hidden_size = en_hidden_size,
                 zh_hidden_size = zh_hidden_size,
                 dropout_p = dropout_p,
                 num_layers = num_layers,
                 bidirectional = bidirectional,
                 weight = weight,
                 zh_maxLength = zh_maxLength,
                 batch_size = batch_size,
                 atten_mode = atten_mode)
    

    train.train(use_cuda=use_cuda, 
                lr = lr, 
                net=net,
                epoches = Epoches, 
                train_loader = train_loader,
                valid_loader = valid_loader,
                print_every = print_every, 
                save_model_every = save_model_every, 
                batch_size = batch_size,
                transformer = tf, 
                agent = agent,
                hyperparameters = hyperparameters,
                tf_ratio=tf_ratio)
    
    train.printPredictsFromDataset2(use_cuda, net, valid_loader, tf, count = 10)
    
    
    
    
    
    
