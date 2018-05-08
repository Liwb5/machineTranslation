import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import random
from Attention import Attention


class Decoder(nn.Module):
    def __init__(self, voc_size, input_size, enc_hidden_size,dec_hidden_size, 
                 dropout_p, maxLength, atten_mode=None, use_cuda=False):
        """
        voc_size: 词典中词的数量
        input_size: 目标语言的词向量维度
        enc_didden_size: encoder的隐藏向量长度
        dec_hidden_size: decoder的隐藏向量长度
        maxLength: 句子最大长度
        """
        super(Decoder, self).__init__()
        
        self.use_cuda = use_cuda
        self.dec_hidden_size = dec_hidden_size
        self.maxLength = maxLength
        self.voc_size = voc_size
        self.atten_mode = atten_mode

        self.embedding = nn.Embedding(num_embeddings = voc_size,
                                      embedding_dim = input_size)
        
        self.lstm_cell = nn.LSTMCell(input_size = input_size,
                                    hidden_size = dec_hidden_size)
        
        #隐藏向量映射到词典大小，数值最大的即为预测结果
        self.hx2voc = nn.Linear(in_features = hidden_size,
                                out_features = voc_size)
        
        self.atten = Attention(use_cuda = use_cuda,
                               mode = atten_mode,
                               enc_hidden_size = enc_hidden_size,
                               dec_hidden_size = dec_hidden_size)
        
        
        
    def forward(self, enc_outputs, enc_hn, enc_cn, gruths, tf_ratio=1, is_eval=False):
        """
        enc_outputs: B * maxLen * enc_hidden_size, 编码器每个cell的隐藏向量
        enc_hn: B * enc_hidden_size, 编码器最后一个cell的hidden state
        enc_cn: B * enc_hidden_size, 编码器最后一个cell的cell state
        gruths: B * tar_maxLen，目标语言的真实答案
        tf_ratio: float, schedule sampling的概率。
        """
        
        #embed -> B * tar_maxLen * dims
        embed = self.embedding(gruths)
        embed = torch.transpose(embed, 0, 1)
        cx = enc_cn.squeeze(0) #3D to 2D
        hx = enc_hn.squeeze(0)
        
        #保存decoder中每个cell的输出结果
        probas = [0 for i in range(embed.size(0))]
        predicts = [0 for i in range(embed.size(0))]
        
        for i in range(embed.size(0)):
            
            if is_eval:
                if i == 0:
                    x = embed[i]
                else:
                    x = self.embedding(predicts[i-1])
            else:
                #判断是否使用上一次预测的结果作为下一次的输入
                if random.random() < tf_ratio:
                    use_tf_ratio = True
                else:
                    use_tf_ratio = False
                    
                if use_tf_ratio or i == 0:
                    x = embed[i]
                else:
                    x = self.embedding(predicts[i-1])
               
            hx, cx = self.lstm_cell(x, (hx, cx))
            
            #attention机制
            if self.atten_mode != None:
                context = self.atten(enc_outputs, hx)
                hx_atten = self.ht_(torch.cat((context, hx), 1))
                hx_atten = F.tanh(hx_atten)
            
            probas[i] = self.hx2voc(hx_atten)        
            
            _, predicts[i] = torch.max(probas[i], 1)
            
            probas[i] = probas[i].view(1, probas[i].size(0), probas[i].size(1))
            predicts[i] = predicts[i].view(1, predicts[i].size(0))
            
            
            
        probas = torch.cat(probas)
        predicts = torch.cat(predicts, 0)
        #probas --> zh_maxLen * B * zh_voc so change it to B* L * zh_voc
        #predicts --> zh_maxLen * B   so change it to B * L
        #predicts 转成data是为了在预测的时候可以使用
        return probas.transpose(0, 1), predicts.contiguous().transpose(0, 1).data.cpu()
        
        
        
        
        
        
        
        
        
        
        
