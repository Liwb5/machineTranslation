
import torch
import h5py
import os

from Folder import Folder
from torch.utils.data import DataLoader
import dataProcess as dp
import transformer
import train
from decoder import Decoder
from encoder import Encoder
from decoder import Decoder
import seq2seq
from hyperboard import Agent


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

use_cuda = torch.cuda.is_available()

sentence_num = 200  #设置数字表示使用一点数据用于测试，设置None表示使用所有数据进行训练

batch_size = 100
en_dims = 512
zh_dims = 512
en_hidden_size = 512
zh_hidden_size = 512
zh_maxLength = 80
lr = 0.01
Epoches = 50
dropout_p = 0.1
print_every = 10 #每多少个batch就print一次
save_model_every = print_every*10000000

hyperparameters = {'lr':lr,
             'dropout_p':dropout_p,
             'en_dims':en_dims,
             'hidden_size':en_hidden_size,
             'batch_size':batch_size}

if __name__ == '__main__':
    #train_folder = Folder('../data/train.h5',is_test=False)
    train_folder = Folder('../data/train3.h5',is_eval = False, num = sentence_num)
    train_loader = DataLoader(train_folder,
                         batch_size=batch_size,
                         num_workers=1,
                         shuffle=False)

    
    inputlang = dp.Lang('en')
    outputlang = dp.Lang('zh')
    inputlang.load('../data/en_dict3.pkl')
    outputlang.load('../data/zh_dict3.pkl')

    tf = transformer.Transformer(inputlang, outputlang)
    
    agent = Agent(address='127.0.0.1',port=5100)
    
    
    
    print(inputlang.name,inputlang.n_words)
    print(outputlang.name,outputlang.n_words)
    
    weight = [1 for i in range(outputlang.n_words)]
    weight[2] = 0
            
    net = seq2seq.Net(use_cuda = use_cuda,
                 en_voc = inputlang.n_words,
                 en_dims = en_dims,
                 zh_voc = outputlang.n_words,
                 zh_dims = zh_dims,
                 en_hidden_size = en_hidden_size,
                 zh_hidden_size = zh_hidden_size,
                 dropout_p = dropout_p,
                 weight = weight,
                 zh_maxLength = zh_maxLength,
                 batch_size = batch_size)
    
    #pre_trained = torch.load('../models/test.model')
    #net.load_state_dict(pre_trained)
    print(net)

    train.train(use_cuda=use_cuda, 
            lr = lr, 
            net=net,
            epoches = Epoches, 
            train_loader=train_loader, 
            print_every = print_every, 
            save_model_every = save_model_every, 
            batch_size = batch_size,
            transformer = tf, 
            agent = agent,
            hyperparameters = hyperparameters)
    
    train.printPredictsFromDataset(use_cuda, net, train_loader, tf, count = 10)
    
    
    
    
    
    