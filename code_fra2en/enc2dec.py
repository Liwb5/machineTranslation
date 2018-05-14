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


from seq2seq.model import DecoderRNN
from seq2seq.model import EncoderRNN
from seq2seq.model import TopKDecoder


class Net(nn.Module):
    def __init__(self, use_cuda, en_voc, en_dims, en_hidden_size, 
                zh_voc, zh_dims, zh_hidden_size, dropout_p, num_layers, bidirectional,
                weight,zh_maxLength, batch_size, atten_mode):

        super(Net, self).__init__()

        self.use_cuda = use_cuda
        self.en_voc = en_voc      #英文单词的个数
        self.en_dims = en_dims    #英文词向量的长度
        self.en_hidden_size = en_hidden_size   #英文词隐藏层向量的长度
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
        """
        self.encoder = Encoder(use_cuda = use_cuda,
                                en_dims = en_dims,
                                en_hidden_size = en_hidden_size,
                                dropout_p = dropout_p,
                                num_layers = num_layers,
                                bidirectional = bidirectional)

        
        self.decoder = Decoder(use_cuda = use_cuda, 
                                zh_voc = zh_voc,
                                zh_dims = zh_dims,
                                zh_hidden_size = zh_hidden_size,
                                dropout_p = dropout_p,
                                batch_size = batch_size,
                                zh_maxLength = zh_maxLength,
                                en_hidden_size = en_hidden_size,
                                atten_mode = atten_mode)
        """
        self.encoder2 = EncoderRNN(vocab_size = en_voc,
                                   max_len = 21,
                                   hidden_size = en_hidden_size,
                                   input_dropout_p = 0,
                                   dropout_p = 0,
                                   n_layers = num_layers,
                                   bidirectional = bidirectional,
                                   rnn_cell = 'lstm',
                                   variable_lengths = True
                                   )
        
        self.decoder2 = DecoderRNN(vocab_size = zh_voc,
                                            max_len = zh_maxLength, 
                                            hidden_size = zh_hidden_size,
                                            sos_id = 0,
                                            eos_id = 1,
                                            n_layers = num_layers,
                                            rnn_cell = 'lstm',
                                            bidirectional = bidirectional,
                                            input_dropout_p = 0,
                                            dropout_p = 0,
                                            use_attention = True)
        
    def order(self, inputs, entext_len):
        """
        order函数将句子的长度按从大到小排序
        inputs: B*en_maxLen. a tensor object
        entext_len: B * 1. the real length of every sentence
        
        return:
        inputs: B * maxLen  tensor
        entext_len: B * 1  tensor
        order_ids:  B * 1  tensor
        """
        #将entext_len按从大到小排序
        sorted_len, sort_ids = torch.sort(entext_len, dim = 0, descending=True)
        
        sort_ids = Variable(sort_ids).cuda() if self.use_cuda else Variable(sort_ids)
        
        inputs = inputs.index_select(0, sort_ids)
        
        _, true_order_ids = torch.sort(sort_ids, 0, descending=False)
        
        #true_order_ids = Variable(true_order_ids).cuda() if self.use_cuda else Variable(true_order_ids)
        
        #排序之后，inputs按照句子长度从大到小排列
        #true_order_ids是原来batch的顺序，因为后面需要将顺序调回来
        return inputs, sorted_len, true_order_ids
    

    def forward(self, entext, zh_gtruths, entext_len,teacher_forcing_ratio = 1, is_eval=False):
        """
        entext: B*en_maxLen 的英文句子，Long tensor
        zh_gtruths： B*zh_maxLen 的中文句子，Long tensor
        entext_len: entext中的每个句子的真实长度
        """

        if self.use_cuda:
            entext = Variable(entext).long().cuda()
            zh_gtruths = Variable(zh_gtruths).long().cuda()
        else:
            entext = Variable(entext).long()
            zh_gtruths = Variable(zh_gtruths).long()
          
        # order函数将句子的长度按从大到小排序
        #entext: varibale, sorted_len: tensor, true_order_ids: variable
        entext, sorted_len, true_order_ids = self.order(entext, entext_len)
        #print(entext.size())
        #print(sorted_len)
        
        #embedding的输入需要是variable
        #en_embed: B * maxLen * en_dim
        en_embedding = self.en_embedding(entext)

        # encoder_outputs --> B * en_maxLen * en_hidden_size
        # encoder_h_n -->  (num_layers * num_directions) * B * en_hidden_size
        #encoder_outputs, encoder_h_n, encoder_c_n = self.encoder(en_embedding, sorted_len)
        #print(encoder_outputs.size(), encoder_h_n.size())
        encoder_outputs, hidden = self.encoder2(entext, list(sorted_len))
        encoder_h_n = hidden[0]
        encoder_c_n = hidden[1]
        #print(encoder_outputs.size(), encoder_h_n.size())
        #换回原先的顺序
        encoder_outputs = encoder_outputs.index_select(0, true_order_ids)
        
        encoder_h_n = encoder_h_n.index_select(1, true_order_ids)
        encoder_c_n = encoder_c_n.index_select(1, true_order_ids)
        
        #logits --> B * L* zh_voc
        #predicts --> B * zh_maxLen  
        """
        dec_outputs, predicts = self.decoder(zh_gtruths, encoder_outputs, encoder_h_n,
                                        encoder_c_n, entext_len,
                                        teacher_forcing_ratio= teacher_forcing_ratio,
                                        is_eval=is_eval)
        """
        #dec_outputs: a seq_len of list of tensor and within which is (batch, zh_voc), 
        dec_outputs, dec_hidden, ret_dict = self.decoder2(inputs=zh_gtruths,
                                                          encoder_hidden = (encoder_h_n, encoder_c_n),
                                                          encoder_outputs = encoder_outputs,
                                                          teacher_forcing_ratio = teacher_forcing_ratio)

        predicts = ret_dict[self.decoder2.KEY_SEQUENCE]
        predicts = torch.cat(predicts, 1).contiguous().data.cpu()
        
        #logits -->B  * zh_maxLen * zh_voc
        #predicts --> B * zh_maxLen
        #return logits, predicts
        return dec_outputs, predicts

    """
    def get_loss(self, logits, labels):
        
        #logits --> B * zh_maxLen * zh_voc
        #labels --> B * zh_maxLen
   
        labels = labels[:,:-1]
        if self.use_cuda:
            labels = Variable(labels).long().cuda()
        else:
            labels = Variable(labels).long()

        logits = logits.contiguous().view(-1, logits.size(-1))
        #logits = logits.data
        labels = labels.contiguous().view(-1)

        loss = torch.mean(self.cost_func(logits, labels))

        return loss
    
    def get_loss(self, dec_outputs, labels):
        
        #labels = labels[:,:-1]
        if self.use_cuda:
            labels = Variable(labels).long().cuda()
        else:
            labels = Variable(labels).long()
        #print(len(dec_outputs))  
        #print(dec_outputs[0].size())
        logits = torch.cat(dec_outputs, 0)#(batch*seq_len, zh_voc)
        #print(logits.size())
        #logits = logits.contiguous().view(-1, logits.size(-1))
        labels = labels.transpose(0,1).contiguous().view(-1)
        
        loss = torch.mean(self.cost_func(logits, labels))
        
        return loss
    """
    def get_loss(self, logits, labels):
        
        if self.use_cuda:
            labels = Variable(labels).long().cuda()
        else:
            labels = Variable(labels).long()
        #labels = labels[:,:-1]
        labels = labels.transpose(0, 1)
        
        for i in range(len(logits)):
            logits[i] = logits[i].contiguous().view(1, logits[i].size(0), logits[i].size(1))
        logits = torch.cat(logits)
        
        logits = logits.contiguous().view(-1, logits.size(-1))
        labels = labels.contiguous().view(-1)
        
        loss = torch.mean(self.cost_func(logits, labels))
        
        return loss
    
    