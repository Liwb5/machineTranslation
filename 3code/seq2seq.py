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
                zh_maxLength, batch_size):

        super(Net, self).__init__()

        self.use_cuda = use_cuda
        self.en_voc = en_voc
        self.en_dims = en_dims
        self.en_hidden_size = en_hidden_size
        self.zh_voc = zh_voc
        self.zh_dims = zh_dims
        self.zh_hidden_size = zh_hidden_size
        self.weight = torch.Tensor(weight)

        self.en_embedding = nn.Embedding(num_embeddings = en_voc,
                                        embedding_dim = en_dims)

        self.zh_embedding = nn.Embedding(num_embeddings = zh_voc,
                                        embedding_dim = zh_dims)

        self.cost_func = nn.CrossEntropyLoss(weight=self.weight)

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
                                en_hidden_size = en_hidden_size)

    def forward(self, inputs, gtruths, inputs_len):
        """
        inputs: B*en_maxLen*en_dims 的list
        gtruths： B*zh_maxLen*zh_dims 的list
        inputs_len: inputs中的每个句子的长度
        """

        if self.use_cuda:
            inputs = Variable(inputs).long().cuda()
            gtruths = Variable(gtruths).long().cuda()
        else:
            inputs = Variable(inputs).long()
            gtruths = Variable(gtruths).long()

        #B * maxLen * hidden_size
        inputs = self.en_embedding(inputs)
        gtruths = self.zh_embedding(gtruths)

        # encoder_outputs --> B * maxLen * en_hidden_size
        encoder_outputs = self.encoder(inputs)

        #logits --> B * L* zh_voc
        #predicts --> B * L   it is not tensor
        logits, predicts = self.decoder(gtruths, encoder_outputs)

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