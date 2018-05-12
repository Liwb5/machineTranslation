import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils

class Encoder(nn.Module):
    def __init__(self, use_cuda, en_dims, en_hidden_size, num_layers,
        dropout_p, bidirectional=False):
        """
        en_dims: 英文词向量的维度
        en_hidden_size: 英文隐藏层的维度

        """
        super(Encoder, self).__init__()

        self.use_cuda = use_cuda

        if bidirectional:
            self.en_hidden_size = int(en_hidden_size/2)
        else:
            self.en_hidden_size = en_hidden_size
        
        
        self.lstm = nn.LSTM(input_size = en_dims,  #输入词向量的维度
                            hidden_size = self.en_hidden_size,  # hx的维度
                            num_layers = num_layers,
                            bias = True,
                            batch_first = False,
                            dropout = dropout_p,
                            bidirectional = bidirectional)#双向lstm

    def forward(self, en_embedding, sent_len):
        """
        en_embedding: B*en_maxLen*en_dims句子长度乘以词向量的长度的variable
        sent_len: B * 1. batch中每个句子的长度 tensor
        
        return:
        unpacked.transpose(0, 1): B * en_maxLen * en_hidden_size
        """
        
        #change en_embedding size to en_maxLen*B*en_dims
        en_embedding = torch.transpose(en_embedding, 0, 1)
        #Packs a Tensor containing padded sequences of variable length.
        packed = rnn_utils.pack_padded_sequence(input = en_embedding,  
                                       lengths = list(sent_len))

        #packed_out: en_maxLen* B * en_hidden_size 
        #last_hs 是一个tuple，(h_n, c_n)
        packed_out, last_hs = self.lstm(packed)
        
        # h_n：(num_layers * num_directions, batch, hidden_size)
        h_n, c_n = last_hs

        unpacked, _ = rnn_utils.pad_packed_sequence(packed_out)
        
        #print('unpacked size:', unpacked.size())
        #在返回之前，先将返回值变回B*en_maxLen_en_hidden_size
        return unpacked.transpose(0, 1), h_n, c_n




