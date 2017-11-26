
import torch
from Folder import Folder
from torch.utils.data import DataLoader
import dataProcess as dp
from transformer import Transformer
from net import Seq2Seq

import train

use_cuda = torch.cuda.is_available()



batch_size = 2
en_dims = 256
zh_dims = 256
en_hidden_size = 256
zh_hidden_size = 256



if __name__ == '__main__':
    #train_folder = Folder('../data/train.h5',is_test=False)
    train_folder = Folder('../data/train.h5',is_test=True, is_eval = False)
    train_loader = DataLoader(train_folder,
                         batch_size=batch_size,
                         num_workers=1,
                         shuffle=False)

    
    inputlang = dp.Lang('en')
    outputlang = dp.Lang('zh')
    inputlang.load('../data/en_input_dict.pkl')
    outputlang.load('../data/zh_output_dict.pkl')

    print(inputlang.name,inputlang.n_words)
    print(outputlang.name,outputlang.n_words)
    
    weight = [1 for i in range(outputlang.n_words)]
    weight[2] = 0

    tf = Transformer(inputlang, outputlang)
    
    net = Seq2Seq(use_cuda = use_cuda,
                 en_voc = inputlang.n_words,
                 en_dims = en_dims,
                 zh_voc = outputlang.n_words,
                 zh_dims = zh_dims,
                 en_hidden_size = en_hidden_size,
                 zh_hidden_size = zh_hidden_size,
                 dropout_p = 0,
                 weight = weight)

    train.train(use_cuda=use_cuda, lr = 0.01, net=net, epoches = 5000, train_loader=train_loader)