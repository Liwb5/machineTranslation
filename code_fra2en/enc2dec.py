import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import random

from encoder import Encoder
from decoder import Decoder


class Net(nn.Module):
    def __init__(self, use_cuda, src_voc, src_dims, src_hidden_size, 
                tar_voc, tar_dims, tar_hidden_size, dropout_p, weight,
                num_layers, bidirectional, tar_maxLen, batch_size, atten_mode):
        """
        src_voc: 源语言词典中词的数量
        src_dims: 源语言词向量的维度
        src_hidden_size: 源语言编码器hidden state的维度
        tar_voc: 目标语言词典中词的数量
        tar_dims: 目标语言词向量的维度
        tar_hidden_size: 目标语言解码器hidden state的维度
        tar_maxLen: 目标语言句子的最大长度
        atten_mode: 注意力机制的模式
        num_layers: RNN 的层数
        """
        super(Net, self).__init__()

        self.use_cuda = use_cuda
        self.src_voc = src_voc      
        self.src_dims = src_dims    
        self.src_hidden_size = src_hidden_size   
        self.tar_voc = tar_voc
        self.tar_dims = tar_dims
        self.tar_hidden_size = tar_hidden_size
        self.weight = torch.Tensor(weight)

        if atten_mode != None:
            print('using attention, attention mode is %s ' % atten_mode)
        else:
            print('not using attention mode.......')


        self.cost_func = nn.CrossEntropyLoss(weight=self.weight)

        self.encoder = Encoder( voc_size = src_voc,
                                input_size = src_dims,
                                hidden_size = src_hidden_size,
                                dropout_p = dropout_p,
                                bidirectional = False,
                                use_cuda = use_cuda)
        
        self.decoder = Decoder( voc_size = tar_voc,
                                input_size = tar_dims,
                                enc_hidden_size = src_hidden_size,
                                dec_hidden_size = tar_hidden_size,
                                dropout_p = dropout_p,
                                maxLength = tar_maxLen,
                                atten_mode = atten_mode,
                                use_cuda = use_cuda)
        
        
    def sorting(self, input_sent, sent_len):
        """
        sorting函数将句子的长度按从大到小排序
        input_sent:  B*en_maxLen. a tensor object
        sent_len: B * 1, the real length of every sentence

        """
        #将entext_len按从大到小排序
        sorted_len, sort_ids = torch.sort(sent_len, dim = 0, descending=True)

        sort_ids = sort_ids.cuda() if self.use_cuda else sort_ids

        input_sent = input_sent.index_select(0, sort_ids)

        """
        input_sent: B * maxLen tensor, 句子真实长度按从大到小排
        sorted_len: B * 1 tensor, 句子长度值从大到小排
        sort_ids: B * 1 tensor, 原先句子的位置下标
        """
        return input_sent, sorted_len, sort_ids


    def forward(self, src_sent, src_len, tar_sent, tf_ratio=0, is_eval=False):
        """
        src_sent: B * src_maxLen, tensor. 每一行是一个源句子所有单词对应的下标
        src_len: B * 1, tensor. 每一行记录一个句子的真实长度
        tar_sent: B * tar_maxLen, tensor. 每一行是一个目标句子的所有单词对应的下标
        tf_ratio: int. schedule sampling的概率
        """

        if self.use_cuda:
            src_sent = src_sent.cuda()
            tar_sent = tar_sent.cuda()
        
        src_sent, sorted_len, sort_ids = self.sorting(src_sent, src_len)

        enc_outputs, enc_hn, enc_cn = self.encoder(src_sent, sorted_len)

        enc_outputs = enc_outputs.index_select(0, sort_ids)
        enc_hn = enc_hn.index_select(1, true_order_ids)
        enc_cn = enc_cn.index_select(1, true_order_ids)

        probas, predicts = self.decoder(enc_outputs, enc_hn, enc_cn,
                                        tar_sent, tf_ratio=tf_ratio, 
                                        is_eval=is_eval)


        return probas, predicts


    def getLoss(self, probas, labels):
        """
        probas: B * tar_maxLen * tar_voc
        labels: B * tar_maxLen
        """

        if self.use_cuda:
            labels = labels.cuda()

        probas = probas.contiguous().view(-1, probas.size(-1))
        labels = labels.contiguous().view(-1)

        loss = self.cost_func(probas, labels)

        return loss




