import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

class Decoder(nn.Module):
    def __init__(self, use_cuda, zh_voc, zh_dims, zh_hidden_size, 
                dropout_p, batch_size, zh_maxLength, en_hidden_size):

        super(Decoder, self).__init__()

        self.use_cuda = use_cuda
        self.zh_hidden_size = zh_hidden_size
        self.batch_size = batch_size
        self.zh_maxLength = zh_maxLength
        self.zh_voc = zh_voc

        self.zh_embedding = nn.Embedding(num_embeddings = zh_voc,
                                        embedding_dim = zh_dims)
        
        self.lstm_cell = nn.LSTMCell(input_size = zh_dims,
                                    hidden_size = zh_hidden_size)


        self.en2zh_size = nn.Linear(in_features = en_hidden_size,
                                    out_features = zh_hidden_size)

        self.hx2zh_voc = nn.Linear(in_features = zh_hidden_size,
                                    out_features = zh_voc)
        
    def last_timestep(self, unpacked, sent_len):
        """
        unpacked: B * maxSentenceLen * en_hidden_size
        sent_len: B*1  the real length of every sentence
        """
        #Index of the last output for each sequence
        idx = (sent_len - 1).view(-1, 1).expand(unpacked.size(0), unpacked.size(2)).unsqueeze(1)
        idx = Variable(idx).long().cuda() if self.use_cuda else Variable(idx).long()
        return unpacked.gather(1, idx).squeeze()

    def forward(self, sent_inputs, hidden_state, sent_len, teacher_forcing_ratio, is_eval = False):
        """
        sent_inputs: B * zh_maxLen * zh_dims的中文句子的variable
        hidden_state: B * maxSentenceLen * en_hidden_size variable 
        sent_len: B * 1  记录每个中文句子的长度
        use_teacher_ratio: decode时使用上次预测的结果作为下次的input的概率
        """
        sent_inputs = self.zh_embedding(sent_inputs)
        
        if self.use_cuda:
            cx = Variable(torch.zeros(self.batch_size, self.zh_hidden_size)).cuda()
        else:
            cx = Variable(torch.zeros(self.batch_size, self.zh_hidden_size))


        #hidden_state = torch.transpose(hidden_state, 0, 1).contiguous()
        sent_inputs = torch.transpose(sent_inputs, 0, 1)
        
        # hx size is B*en_hidden_size
        #hx = hidden_state[-1].view(hidden_state.size(1), hidden_state.size(2))
        
        hx = self.last_timestep(hidden_state.contiguous(), sent_len)

        if self.use_cuda:
            logits = Variable(torch.zeros(sent_inputs.size(0), self.batch_size, self.zh_voc)).cuda()
            predicts = Variable(torch.zeros(sent_inputs.size(0), self.batch_size)).long().cuda()
        else:
            logits = Variable(torch.zeros(sent_inputs.size(0), self.batch_size, self.zh_voc))
            predicts = Variable(torch.zeros(sent_inputs.size(0), self.batch_size)).long()
  
        
        for i in range(sent_inputs.size(0)):
            
            if is_eval:
                if i == 0:
                    inputs_x = sent_inputs[i]
                else:
                    inputs_x = self.zh_embedding(predicts[i-1])
            else:
                #判断是否使用上一次预测的结果作为下一次的输入
                #将这个与放在for循环外面则是整个句子预测一次，放在这里则是每个词都预测一次
                if random.random() < teacher_forcing_ratio:
                    use_teacher_forcing = True 
                else:
                    use_teacher_forcing = False
                use_teacher_forcing = True
                if use_teacher_forcing or i == 0:
                    inputs_x = sent_inputs[i]
                    #print('i==%d used:'% i,inputs_x.size())
                else:
                    inputs_x = self.zh_embedding(predicts[i-1])
                    #print('i=%d unused:'% i,inputs_x.size())

            hx, cx = self.lstm_cell(inputs_x,(hx, cx))

            logits[i] = self.hx2zh_voc(hx)

            _, predicts[i] = torch.max(logits[i], 1)
            

        #logits --> zh_maxLen * B * zh_voc so change it to B* L * zh_voc
        #predicts --> zh_maxLen * B   so change it to B * L
        #predicts 转成data是为了在预测的时候可以使用
        return logits.contiguous().transpose(0, 1), predicts.contiguous().transpose(0, 1).data.cpu()



