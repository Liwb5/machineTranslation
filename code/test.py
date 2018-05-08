import torch
import h5py
import os

from Dataset import Dataset
from torch.utils.data import DataLoader
import dataProcess as dp
import transformer
import train
from decoder import Decoder
from encoder import Encoder
from decoder import Decoder
import seq2seq
from hyperboard import Agent


os.environ["CUDA_VISIBLE_DEVICES"] = '1'

use_cuda = torch.cuda.is_available()

sentence_num = 200  #设置数字表示使用一点数据用于测试，设置None表示使用所有数据进行训练

atten_mode = 'general'  #None 表示不使用attention，general表示使用general模式
tf_ratio = None   #测试的时候是1，如果为None表示tf_ratio随着时间变小

batch_size = 200
en_dims = 256
zh_dims = 256
en_hidden_size = 256
zh_hidden_size = 256
zh_maxLength = 80
lr = 0.01
Epoches = 50
dropout_p = 0.1
print_every = 1 #每多少个batch就print一次
save_model_every = batch_size*100000#设置多少个batch就保存一次模型

hyperparameters = {'lr':lr,
             'dropout_p':dropout_p,
             'en_dims':en_dims,
             'hidden_size':en_hidden_size,
             'batch_size':batch_size,
             'dropout_p': dropout_p}

print(hyperparameters)
if __name__ == '__main__':
    path = os.path.dirname(__file__) #获得本文件所在的目录
    if path != "":
        os.chdir(path) #将当前路径设置为本文件所在的目录，方便下面读取文件。
    
    #加载数据，为了可以使用dataLoader批量加载数据，需要定义一个Dataset类，
    #按照pytorch的说明，定义好几个必要的函数后就可以使用dataLoader加载了，详情看Dataset文件。
    """
    trainDataset = Dataset('../dataAfterProcess/train3.h5',is_eval = False, num = sentence_num)
    train_loader = DataLoader(trainDataset,
                         batch_size=batch_size,
                         num_workers=1,#多进程，并行加载
                         shuffle=False)
    """
    validDataset =  Dataset('../dataAfterProcess/valid3.h5', is_eval = False, num = 100)
    valid_loader = DataLoader(validDataset,
                     batch_size = 50,
                     num_workers = 1,
                     shuffle = False)
    
    #加载两个语言库
    inputlang = dp.Lang('en')
    outputlang = dp.Lang('zh')
    inputlang.load('../dataAfterProcess/en_dict3.pkl')
    outputlang.load('../dataAfterProcess/zh_dict3.pkl')


    tf = transformer.Transformer(inputlang, outputlang)
    
    #agent = Agent(address='127.0.0.1',port=5101)
    
    
    
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
                 batch_size = batch_size,
                 atten_mode = atten_mode)
    
    pre_trained = torch.load('../models/lr0.010_BS500_tForce0.741_BLEU4.807_steps0.model')
    net.load_state_dict(pre_trained)
    print(net)
    
    #bleu_score = train.getBLEU(use_cuda, valid_loader, net, transformer)
    """
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
    """
    train.printPredictsFromDataset(use_cuda, net, valid_loader, tf, count = 10)
    
    
    
    
    
    
