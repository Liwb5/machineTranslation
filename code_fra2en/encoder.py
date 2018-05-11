import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

class Encoder(nn.Module):
    def __init__(self, voc_size, input_size, hidden_size, dropout_p, 
                 bidirectional=False, use_cuda=False):
        """
        voc_size: 词典中词的个数
        input_size: 输入词向量的维度
        hidden_size: 隐藏层的维度

        """
        super(Encoder, self).__init__()
        self.use_cuda = use_cuda
        
        #encoder的embedding
        self.embedding = nn.Embedding(num_embeddings = voc_size,
                                        embedding_dim = input_size)

        if bidirectional:
            hidden_size = int(hidden_size/2)

        self.lstm = nn.LSTM(input_size = input_size,  #输入词向量的维度
                            hidden_size = hidden_size,  # hx的维度
                            num_layers = 1,
                            bias = True,
                            batch_first = False,
                            dropout = dropout_p,
                            bidirectional = bidirectional)#双向lstm


    def forward(self, sent, sentLen):
        """
        params:
            sent: B*maxLen batch中的每个句子对应的下标
            sentLen: B*1 batch中每个句子的真实长度
        """

        #embed -> B * maxLen * dims
        embed = self.embedding(sent)
        
        #change embedding size to maxLen*B*dims
        embed = torch.transpose(embed, 0, 1)

        #Packs a Tensor containing padded sequences of variable length.
        packed = rnn_utils.pack_padded_sequence(input = embed,  
                                       lengths = list(sentLen))

        #packed_out: maxLen* B * hidden_size 
        #last_hs 是一个tuple，(h_n, c_n)
        packed_out, last_hs = self.lstm(packed)
        
        # h_n：(num_layers * num_directions, batch, hidden_size)
        h_n, c_n = last_hs

        unpacked, _ = rnn_utils.pad_packed_sequence(packed_out)
        
        #print('unpacked size:', unpacked.size())
        #在返回之前，先将返回值变回B*maxLen*hidden_size
        return unpacked.transpose(0, 1), h_n, c_n
    
    
    
    
    
    
    
    
    
    
    
    