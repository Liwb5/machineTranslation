import torch
import torch.nn as nn
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, use_cuda, en_dims, en_hidden_size, 
        dropout_p, bidirectional=False):
        """
        en_dims: 输入词向量的维度
        en_hidden_size: 输入隐藏层的维度

        """
        super(Encoder, self).__init__()

        self.use_cuda = use_cuda

        if bidirectional:
            en_hidden_size = en_hidden_size/2

        self.lstm = nn.LSTM(input_size = en_dims,  #输入词向量的维度
                            hidden_size = en_hidden_size,  # hx的维度
                            num_layers = 1,
                            bias = True,
                            batch_first = False,
                            dropout = dropout_p,
                            bidirectional = bidirectional)#双向lstm

    def forward(self, sent_inputs, sent_len = None):
        """
        sent_inputs: B*en_maxLen*en_dims句子长度乘以词向量的长度的variable
        sent_len: batch中每个句子的长度
        """
        #packed = rnn_utils.pack_padded_sequence(input = sent_inputs,  
        #                               lengths = list(sent_len))
        
        #change sent_inputs size to en_maxLen*B*en_dims
        sent_inputs = torch.transpose(sent_inputs, 0, 1)

        #result: en_maxLen* B * en_hidden_size 
        result, _ = self.lstm(sent_inputs)

        
        return result.transpose(0, 1)




