import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json
import random
import torch.nn.utils.rnn as rnn_utils
from encoder import Encoder
from decoder import Decoder



class Net(nn.Module):
    def __init__(self, use_cuda, en_voc, en_dims, en_hidden_size, 
                zh_voc, zh_dims, zh_hidden_size, dropout_p, weight,
                zh_maxLength, batch_size, atten_mode):

        super(Net, self).__init__()

        self.use_cuda = use_cuda
        self.en_voc = en_voc      #英文单词的个数
        self.en_dims = en_dims    #英文词向量的长度
        self.en_hidden_size = en_hidden_size   
        self.zh_voc = zh_voc
        self.zh_dims = zh_dims
        self.zh_hidden_size = zh_hidden_size
        self.weight = torch.Tensor(weight)
        
        if atten_mode != None:
            print('using attention, attention mode is %s \n' % atten_mode)
        else:
            print('not using attention mode.......')

        #encoder的embedding
        self.en_embedding = nn.Embedding(num_embeddings = en_voc,
                                        embedding_dim = en_dims)

        

        self.cost_func = nn.CrossEntropyLoss(weight=self.weight)

        #encoder的embedding是放在了Net类这里
        self.encoder = Encoder(use_cuda = use_cuda,
                                en_dims = en_dims,
                                en_hidden_size = en_hidden_size,
                                dropout_p = dropout_p,
                                bidirectional = False)

        self.decoder = Decoder(use_cuda = use_cuda, 
                                zh_voc = zh_voc,
                                zh_dims = zh_dims,
                                zh_hidden_size = zh_hidden_size,
                                dropout_p = dropout_p,
                                batch_size = batch_size,
                                zh_maxLength = zh_maxLength,
                                en_hidden_size = en_hidden_size,
                                atten_mode = atten_mode)
        
        
    def order(self, inputs, inputs_len):
        """
        order函数将句子的长度按从大到小排序
        inputs: B*en_maxLen. a tensor object
        inputs_len: B * 1. the real length of every sentence
        
        return:
        inputs: B * maxLen  tensor
        inputs_len: B * 1  tensor
        order_ids:  B * 1  tensor
        """
        #将inputs_len按从大到小排序
        inputs_len, sort_ids = torch.sort(inputs_len, dim = 0, descending=True)
        
        #sort_ids = Variable(sort_ids).cuda() if self.use_cuda else Variable(sort_ids)
        
        inputs = inputs.index_select(0, sort_ids.view(-1))#sort_ids 是B*1的矩阵，要转换成vector
        
        #似乎这一步没有必要
        #_, true_order_ids = torch.sort(sort_ids, 0, descending=False)
        
        #true_order_ids = Variable(true_order_ids).cuda() if self.use_cuda else Variable(true_order_ids)
        
        #排序之后，inputs按照句子长度从大到小排列
        #true_order_ids是原来batch的顺序，因为后面需要将顺序调回来
        return inputs, inputs_len, order_ids
    

    def forward(self, inputs, gtruths, inputs_len,teacher_forcing_ratio = 1, is_eval=False):
        """
        inputs: B*en_maxLen 的Long tensor
        gtruths： B*zh_maxLen 的Long tensor
        inputs_len: inputs中的每个句子的真实长度
        """

        #似乎这一步没有什么必要
        """
        if self.use_cuda:
            inputs = Variable(inputs).long().cuda()
            gtruths = Variable(gtruths).long().cuda()
        else:
            inputs = Variable(inputs).long()
            gtruths = Variable(gtruths).long()
        """    
        # order函数将句子的长度按从大到小排序
        inputs, sorted_len, true_order_ids = self.order(inputs, inputs_len)

        #embedding的输入需要是variable
        inputs = Variable(inputs).cuda() if self.use_cuda else Variable(inputs)
        #inputs: B * maxLen * en_dim
        inputs = self.en_embedding(inputs)
        

        # encoder_outputs --> B * en_maxLen * en_hidden_size
        # encoder_h_n -->  (num_layers * num_directions) * B * en_hidden_size
        encoder_outputs, encoder_h_n = self.encoder(inputs, sorted_len)
        
        #换回原先的顺序
        encoder_outputs = encoder_outputs.index_select(0, true_order_ids)

        #logits --> B * L* zh_voc
        #predicts --> B * L   it is not tensor
        logits, predicts = self.decoder(gtruths, encoder_outputs, encoder_h_n, inputs_len,
                    teacher_forcing_ratio= teacher_forcing_ratio, is_eval=is_eval)

        #logits -->B  * zh_maxLen * zh_voc
        #predicts --> B * zh_maxLen
        return logits, predicts


    def get_loss(self, logits, labels):
        """
        logits --> zh_maxLen * B * zh_voc
        labels --> B * zh_maxLen * 1
        """
        if self.use_cuda:
            labels = Variable(labels).long().cuda()
            #logits = Variable(logits).long().cuda()
        else:
            labels = Variable(labels).long()
            #logits = Variable(logits).long()

        #labels = labels.transpose(0, 1)

        logits = logits.contiguous().view(-1, logits.size(-1))
        #logits = logits.data
        labels = labels.contiguous().view(-1)

        loss = torch.mean(self.cost_func(logits, labels))

        return loss