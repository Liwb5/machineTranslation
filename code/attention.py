import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils


class Attention(nn.Module):
    def __init__(self, use_cuda, mode, en_hidden_size, zh_hidden_size):
        """
        mode: attention有不同的做法可以选择
        en_hidden_size: encoder 的hidden size
        """
        super(Attention, self).__init__()

        self.use_cuda = use_cuda
        self.atten_mode = mode
        self.en_hidden_size = en_hidden_size
        self.zh_hidden_size = zh_hidden_size

        #define layers
        if self.atten_mode == 'general':
            self.attention = nn.Linear(self.en_hidden_size, 
                                        self.zh_hidden_size)
        elif self.atten_mode == 'concat':
            self.attention = nn.Linear(self.zh_hidden_size + self.en_hidden_size, self.zh_hidden_size)

   

    def forward2(self, hx, encoder_outputs):
        """
        hx: B * zh_hidden_size
        encoder_outputs: B * maxLen * en_hidden_size
        """
        energies = self._atten_weight(hx, encoder_outputs)
        #其实就是返回一个权重向量
        return F.softmax(energies, dim=1).unsqueeze(1)    
   
        
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
    
    def _atten_weight(self, hx, encoder_outputs):
        """
        from: Effective Approaches to Attention-based Neural Machine Translation
        hx: B * zh_hidden_size
        encoder_outputs: B * maxLen * en_hidden_size
        """
        encoder_outputs = encoder_outputs.transpose(0, 1)
        
        #change hx to maxLen * B * zh_hidden_size
        hx = hx.expand(encoder_outputs.size(0), hx.size(0), hx.size(1))
        
        
        if self.mode == 'dot':
            #energies = torch.mm()
            pass
        elif self.mode == 'general':
            energies = self.attention(encoder_outputs)
            energies = torch.sum(hx*energies, 2)
        elif self.mode == 'concat':
            pass
        else:
            print('the attention mode is not right.')
           
        #change to B * maxLen 
        energies = energies.transpose(0, 1)
        return energies
