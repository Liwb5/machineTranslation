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

        self.zh_embedding = nn.Embedding(num_embeddings = zh_voc,
                                        embedding_dim = zh_dims)
        
        self.lstm_cell = nn.LSTMCell(input_size = zh_dims,
                                    hidden_size = zh_hidden_size)


        self.en2zh_size = nn.Linear(in_features = en_hidden_size,
                                    out_features = zh_hidden_size)

        self.hx2zh_voc = nn.Linear(in_features = zh_hidden_size,
                                    out_features = zh_voc)

    def forward(self, sent_inputs, hidden_state, sent_len = None, is_eval = False):
        """
        sent_inputs: B * zh_maxLen * zh_dims的中文句子的variable
        hidden_state: B * en_maxLen * en_hidden_size 
        sent_len: B * 1  记录每个中文句子的长度
        """
        sent_inputs = self.zh_embedding(sent_inputs)
        
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
            logits = Variable(torch.zeros(sent_inputs.size(0), self.batch_size, self.zh_voc)).cuda()
            predicts = Variable(torch.zeros(sent_inputs.size(0), self.batch_size)).long().cuda()
        else:
            logits = Variable(torch.zeros(sent_inputs.size(0), self.batch_size, self.zh_voc))
            predicts = Variable(torch.zeros(sent_inputs.size(0), self.batch_size)).long()

        logits = [0 for i in range(sent_inputs.size(0)-1)]
        predicts = [0 for i in range(sent_inputs.size(0)-1)]
        
        for i in range(sent_inputs.size(0)-1):
            
            if is_eval:
                if i == 0:
                    inputs_x = sent_inputs[0]
                else:
                    inputs_x = self.zh_embedding(predicts[i-1].transpose(0,1))
            else:
                inputs_x = sent_inputs[i]

            hx, cx = self.lstm_cell(inputs_x,(hx, cx))

            logits[i] = self.hx2zh_voc(hx)

            _, predicts[i] = torch.max(logits[i], 1)
            
            logits[i] = logits[i].view(1, logits[i].size(0), logits[i].size(1))
            predicts[i] = predicts[i].view(1, predicts[i].size(0))
            
        predicts = torch.cat(predicts, 0)
        predicts = torch.transpose(predicts, 0, 1)
            
        if is_eval:     
            #print(predicts)
            pass
        #print(sent_inputs[0].size())

        #logits --> zh_maxLen * B * zh_voc so change it to B* L * zh_voc
        #predicts --> zh_maxLen * B   so change it to B * L
        #predicts 转成data是为了在预测的时候可以使用
        return torch.cat(logits).transpose(0, 1), predicts.data.cpu()



