import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, use_cuda, zh_voc, zh_dims, zh_hidden_size, 
                dropout_p, batch_size, zh_maxLength, en_hidden_size):

        super(Decoder, self).__init__()

        self.use_cuda = use_cuda
        self.zh_hidden_size = zh_hidden_size
        self.batch_size = batch_size
        self.zh_maxLength = zh_maxLength
        self.zh_voc = zh_voc

        self.lstm_cell = nn.LSTMCell(input_size = zh_dims,
                                    hidden_size = zh_hidden_size)


        self.en2zh_size = nn.Linear(in_features = en_hidden_size,
                                    out_features = zh_hidden_size)

        self.hx2zh_voc = nn.Linear(in_features = zh_hidden_size,
                                    out_features = zh_voc)

    def forward(self, sent_inputs, hidden_state, sent_len = None):
        """
        sent_inputs: B * zh_maxLen * zh_dims的中文句子的variable
        hidden_state: B * en_maxLen * en_hidden_size 
        sent_len: B * 1  记录每个中文句子的长度
        """
        if self.use_cuda:
            cx = Variable(torch.zeros(self.batch_size, self.zh_hidden_size)).cuda()
        else:
            cx = Variable(torch.zeros(self.batch_size, self.zh_hidden_size))


        hidden_state = torch.transpose(hidden_state, 0, 1)
        sent_inputs = torch.transpose(sent_inputs, 0, 1)
        # hx size is B*en_hidden_size
        hx = hidden_state[-1].view(hidden_state.size(1), hidden_state.size(2))

        #change the hx size to B * zh_hidden_size
        if hidden_state.size(2) != self.zh_hidden_size:  
            hx = self.en2zh_size(hx)

        if self.use_cuda:
            logits = Variable(torch.zeros(self.zh_maxLength, self.batch_size, self.zh_voc)).cuda()
            predicts = Variable(torch.zeros(self.zh_maxLength, self.batch_size)).cuda()
        else:
            logits = Variable(torch.zeros(self.zh_maxLength, self.batch_size, self.zh_voc))
            predicts = Variable(torch.zeros(self.zh_maxLength, self.batch_size))
        
        for i in range(self.zh_maxLength):
            hx, cx = self.lstm_cell(sent_inputs[i],(hx, cx))

            logits[i] = self.hx2zh_voc(hx)

            _, predicts[i] = torch.max(logits[i], 1)

        #logits --> zh_maxLen * B * zh_voc so change it to B* L * zh_voc
        #predicts --> zh_maxLen * B   so change it to B * L
        #predicts 转成data是为了在预测的时候可以使用
        return logits.transpose(0, 1), predicts.transpose(0, 1).data.cpu()



