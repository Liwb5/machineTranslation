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
        self.mode = mode
        self.en_hidden_size = en_hidden_size
        self.zh_hidden_size = zh_hidden_size

        #define layers
        if self.mode == 'general':
            self.attention = nn.Linear(self.en_hidden_size, 
                                        self.zh_hidden_size)
        elif self.mode == 'concat':
            self.attention = nn.Linear(self.zh_hidden_size + self.en_hidden_size, self.zh_hidden_size)


    def forward(self, hx, encoder_outputs):
        """
        hx: B * zh_hidden_size
        encoder_outputs: B * maxLen * en_hidden_size
        """

        seq_len = encoder_outputs.size(1)

        # energies --> maxLen * B
        if self.use_cuda:
            energies = Variable(torch.zeros(seq_len, encoder_outputs.size(0))).cuda()
        else:
            energies = Variable(torch.zeros(seq_len, encoder_outputs.size(0)))

        encoder_outputs = encoder_outputs.transpose(0, 1)

        for i in range(seq_len):
            energies[i] = self._score(hx, encoder_outputs[i])

        #change to B * maxLen 
        energies = energies.transpose(0, 1)
        
        #其实就是返回一个权重向量
        return F.softmax(energies).unsqueeze(1)

    def _score(self, hx, encoder_output):

        if self.mode == 'dot':
            energy = torch.sum(hx * encoder_output, 1) #按行点乘

        elif self.mode == 'general':
            energy = self.attention(encoder_output)
            energy = torch.sum(hx * energy, 1)
            #print(energy)
        elif self.mode == 'concat':#has not finished concat mode
            #print(torch.cat((hx, encoder_output))
            energy = self.attention(torch.cat((hx, encoder_output), 1)) #按行拼接
            
        return energy
