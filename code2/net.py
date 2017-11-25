import numpy as np 
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json
import random
import torch.nn.utils.rnn as rnn_utils


class Encoder(nn.Module):
    def __init__(self, use_cuda, en_dims, en_hidden_size, dropout_p, bidirectional=False):

        super(Encoder, self).__init__()

        self.use_cuda = use_cuda

        self.lstm = nn.LSTM(input_size = en_dims,  #输入词向量的维度
                            hidden_size = en_hidden_size,  # hx的维度
                            num_layers = 1,
                            bias = True,
                            batch_first = False,
                            dropout = dropout_p, 
                            bidirectional = bidirectional)#双向lstm

    #inputs 是Len*Batch*en_dims   inputs_len 是记录每个句子的长度(长度是降序的)，它的长度就是batch_size*1大小。
    def forward(self, inputs, inputs_len=None):
        #packed = rnn_utils.pack_padded_sequence(input = inputs,  
        #                               lengths = list(inputs_len))

        #packed_out 是Len*Batch*(en_hidden_size*2)
        #packed_out, _ = self.lstm(packed)

        unpacked, _ = self.lstm(inputs)
        
        #经过unpack之后，unpacked是 maxLen*batch*en_hidden_size*num_bidirection
        #注意maxLen是这个batch中句子最长的那个，长度小的句子会补0
        #num_bidirection 由bidirectional这个参数决定，双向就是hidden_size会翻倍
        #unpacked, _ = rnn_utils.pad_packed_sequence(packed_out)
        
        return unpacked

class Decoder(nn.Module):
    def __init__(self, use_cuda, zh_dims, zh_hidden_size, en_hidden_size, zh_voc):
        super(Decoder, self).__init__()

        self.use_cuda = use_cuda

        self.zh_hidden_size = zh_hidden_size

        self.lstm_cell = nn.LSTMCell(input_size = zh_dims,
                                    hidden_size = zh_hidden_size)

        #加入attention，由于attention有自己的维度，所以需要将hidden_state的维度转换到atten_vec_size
#         self.atten_ws = nn.Linear(in_features = zh_hidden_size,
#                                 out_features = atten_vec_size)
        
        self.en2zh_size = nn.Linear(in_features = en_hidden_size, 
                                    out_features = zh_hidden_size)

        self.hx2zh_voc = nn.Linear(in_features = zh_hidden_size,
                                    out_features = zh_voc)

    #encoder_outputs是maxLen*batch*(en_hidden_size*num_bidirection)
    #inputs就是gtruths，即正确的翻译,batch*zh_dims
    def forward(self, inputs, encoder_outputs):

        hx = encoder_outputs[-1] #只取最后一个输出

        #hx = self.en2zh_size(hx)# batch*zh_hidden_size

        if self.use_cuda:
            #这里是size(0)，感觉不是呀
            #hx = Variable(torch.zeros(encoder_outputs.size(0), self.zh_hidden_size)).cuda()
            cx = Variable(torch.zeros(encoder_outputs.size(1), self.zh_hidden_size)).cuda()
        else:
            #hx = Variable(torch.zeros(encoder_outputs.size(0), self.zh_hidden_size))
            cx = Variable(torch.zeros(encoder_outputs.size(1), self.zh_hidden_size))

        logits = [0 for i in range(inputs.size(0))]
        predicts = [0 for i in range(inputs.size(0))]
        outputs = Variable(torch.zeros(inputs.size(0), encoder_outputs.size(1), self.zh_hidden_size))
        for i in range(inputs.size(0)):
            #print(inputs[i])
            hx, cx = self.lstm_cell(inputs[i], (hx, cx))
            logits[i] = self.hx2zh_voc(hx)#batch* zh_voc ==> logits[i]
            _, predicts[i] = torch.max(logits[i], 1)#我们要的是最大值对应的下标而已，所以第一个输出不要了

            logits[i] = logits[i].view(1, logits[i].size(0), logits[i].size(1))
            predicts[i] = predicts[i].view(1, predicts[i].size(0))
            
        #转置之后才是2*80. 不取data的话，predicts是variable，似乎无法将其翻译成中文，
        return torch.cat(logits).transpose(0,1), torch.cat(predicts).transpose(0, 1).data.cpu()




class Seq2Seq(nn.Module):
    def __init__(self, use_cuda, en_voc, en_dims, zh_voc, zh_dims, en_hidden_size,
                zh_hidden_size, dropout_p):
        super(Seq2Seq, self).__init__()

        self.use_cuda = use_cuda

        self.en_embedding = nn.Embedding(num_embeddings = en_voc,
                                        embedding_dim = en_dims)

        self.zh_embedding = nn.Embedding(num_embeddings = zh_voc,
                                        embedding_dim = zh_dims)

        self.cost_func = nn.CrossEntropyLoss()

        self.encoder = Encoder(use_cuda = use_cuda,
                                en_dims = en_dims,
                                en_hidden_size = en_hidden_size,
                                dropout_p = dropout_p)

        self.decoder = Decoder(use_cuda = use_cuda, 
                                zh_dims = zh_dims,
                                zh_hidden_size = zh_hidden_size,
                                en_hidden_size = en_hidden_size,
                                zh_voc = zh_voc)

    #句子就以原始形式作为参数
    def forward(self, inputs, gtruths, inputs_len):
        
        inputs = inputs.transpose(0, 1)
        gtruths = gtruths.transpose(0, 1)
        
        
        if self.use_cuda:
            inputs = Variable(inputs).long().cuda()
            gtruths = Variable(gtruths).long().cuda()
        else:
            inputs = Variable(inputs).long()
            gtruths = Variable(gtruths).long()
            
        inputs = self.en_embedding(inputs)
        encoder_outputs = self.encoder(inputs)

        gtruths = self.zh_embedding(gtruths)
        #print(type(encoder_outputs))
        logits, predicts = self.decoder(gtruths, encoder_outputs)

        return logits, predicts

    def get_loss(self, logits, labels):
        if self.use_cuda:
            labels = Variable(labels).long().cuda()
        else:
            labels = Variable(labels).long()
            
        labels = labels.transpose(0, 1)

        logits = logits.contiguous().view(-1, logits.size(-1))
        labels = labels.contiguous().view(-1)

        loss = torch.mean(self.cost_func(logits, labels))

        return loss
