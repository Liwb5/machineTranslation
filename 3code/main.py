
import torch
import h5py

from Folder import Folder
from torch.utils.data import DataLoader
import dataProcess as dp
import transformer
import train
from decoder import Decoder
from encoder import Encoder
from decoder import Decoder
import seq2seq


use_cuda = torch.cuda.is_available()


batch_size = 2
en_dims = 256
zh_dims = 256
en_hidden_size = 256
zh_hidden_size = 256
zh_maxLength = 80
lr = 0.01
Epoches = 30
dropout_p = 0.1

if __name__ == '__main__':
    #train_folder = Folder('../data/train.h5',is_test=False)
    train_folder = Folder('../data/train3.h5',is_test=True, is_eval = False)
    train_loader = DataLoader(train_folder,
                         batch_size=batch_size,
                         num_workers=1,
                         shuffle=False)

    
    inputlang = dp.Lang('en')
    outputlang = dp.Lang('zh')
    inputlang.load('../data/en_dict3.pkl')
    outputlang.load('../data/zh_dict3.pkl')

    tf = transformer.Transformer(inputlang, outputlang)
    
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

    train.train(use_cuda=use_cuda, lr = lr, net=net, epoches = Epoches, 
                 train_loader=train_loader, print_every = 20,batch_size = batch_size,
                transformer = tf)
    
    train.evaluateFromDataset(use_cuda, net, train_loader, tf)
    
    
    
    
    
    