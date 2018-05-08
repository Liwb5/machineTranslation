import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

class Attention(nn.Module):
    def __init__(self, atten_mode, src_hidden_size, tar_hidden_size, use_cuda=False):
        """
        atten_mode: attention 有不同的做法可以选择
        src_hidden_size: 源语言的隐藏层向量的维度
        tar_hidden_size: 目标语言的隐藏层向量的维度
        """
        
        super(Attention, self).__init__()
        
        self.use_cuda = use_cuda
        self.atten_mode = atten_mode
        self.src_hidden_size = src_hidden_size
        self.tar_hidden_size = tar_hidden_size

        
        #这部分有疑问
        if self.atten_mode == 'general':
            self.attention = nn.Linear(src_hidden_size, 
                                       tar_hidden_size)
        elif self.atten_mode == 'concat':
            self.attention = nn.Linear(tar_hidden_size+src_hidden_size,
                                       tar_hidden_size)
            
        
    def forward(self, enc_outputs, hx):
        """
        enc_outputs: B * src_maxLen * src_hidden_size 
        hx: B * tar_hidden_size 
        """
        score = self.score_(enc_outputs, hx)
        # at: B * maxLen
        at = F.softmax(score, dim=1).unsqueeze(1)
        
        # ct: B * src_hidden_size
        ct = at.bmm(enc_outputs).squeeze(1)
        
        return ct
            
    def score_(self, enc_outputs, hx):
        """
        enc_outputs: B * src_maxLen * src_hidden_size 
        hx: B * tar_hidden_size 
        """
        #enc_outputs_T: src_maxLen * B * src_hidden_size
        #enc_outputs_T = enc_outputs.transpose(0, 1)
            
        # hx_: src_maxLen * B * tar_hidden_size
        #hx_ = hx.expand(enc_outputs_T.size(0), hx.size(0), hx.size(1))
            
        if self.atten_mode == 'dot':
            #score: B * maxLen
            score = torch.matmul(enc_outputs, hx.unsqueeze(2))
            score = score.squeeze(2).contiguous()
            #score = torch.sum(hx_*enc_outputs_T, 2).transpose(0,1)
        elif self.atten_mode == 'general':
            score = torch.matmul(self.attention(enc_outputs), hx.unsqueeze(2))
            score = score.squeeze(2)
                
        elif self.atten_mode == 'concat':
            pass
        
        #score: B * maxLen
        return score                










