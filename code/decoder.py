import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import attention


class Decoder(nn.Module):
    def __init__(self, use_cuda, zh_voc, zh_dims, zh_hidden_size, 
                dropout_p, batch_size, zh_maxLength, en_hidden_size, atten_mode=None):

        super(Decoder, self).__init__()

        self.use_cuda = use_cuda
        self.zh_hidden_size = zh_hidden_size
        self.batch_size = batch_size
        self.zh_maxLength = zh_maxLength
        self.zh_voc = zh_voc
        self.atten_mode = atten_mode

        self.zh_embedding = nn.Embedding(num_embeddings = zh_voc,
                                        embedding_dim = zh_dims)
        
        self.lstm_cell = nn.LSTMCell(input_size = zh_dims,
                                    hidden_size = zh_hidden_size)


        self.en2zh_size = nn.Linear(in_features = en_hidden_size,
                                    out_features = zh_hidden_size)

        self.hx2zh_voc = nn.Linear(in_features = zh_hidden_size,
                                    out_features = zh_voc)
        
        self.atten = attention.Attention(use_cuda = use_cuda,
                              mode = atten_mode,
                              en_hidden_size = en_hidden_size,
                              zh_hidden_size = zh_hidden_size)
        
        self.ht_ = nn.Linear(in_features = zh_hidden_size*2,
                            out_features = zh_hidden_size)
        
    def last_timestep(self, unpacked, sent_len):
        """
        unpacked: B * maxLen * en_hidden_size
        sent_len: B*1  the real length of every sentence
        """
        #Index of the last output for each sequence
        idx = (sent_len - 1).view(-1, 1).expand(unpacked.size(0), unpacked.size(2)).unsqueeze(1)
        idx = Variable(idx).long().cuda() if self.use_cuda else Variable(idx).long()
        return unpacked.gather(1, idx).squeeze()

    def forward(self, zh_gruths, encoder_hs, encoder_h_n, encoder_c_n, 
                entext_len, teacher_forcing_ratio, is_eval = False):
        """
        zh_gruths: B * zh_maxLen 的中文句子的variable
        encoder_hs: B * en_maxLen * en_hidden_size variable. the hidden state of encoder 
        encoder_h_n: (num_layers * num_directions) * B * en_hidden_size  encoder输出的最后一个隐藏状态
        encoder_c_n: (num_layers * num_directions) * B * en_hidden_size  encoder输出的最后一个隐藏状态
        entext_len: B * 1  记录每个英文句子的长度, tensor
        use_teacher_ratio: decode时使用上次预测的结果作为下次的input的概率
        is_eval: 设置训练阶段还是测试阶段，False表示训练阶段。True表示测试阶段
        """
            
        #zh_gruths: B * zh_maxLen * zh_dim
        zh_embedding = self.zh_embedding(zh_gruths)
        """
        #可以尝试使用encoder的cn
        if self.use_cuda:
            cx = Variable(torch.zeros(zh_embedding.size(0), self.zh_hidden_size)).cuda()
            #hx = Variable(torch.zeros(zh_embedding.size(0), encoder_hs.size(2))).cuda()
        else:
            cx = Variable(torch.zeros(zh_embedding.size(0), self.zh_hidden_size))
        """
        cx = encoder_c_n.squeeze(0)

        # hx size is B*en_hidden_size
        #hx = encoder_hs[-1].view(encoder_hs.size(0), encoder_hs.size(2))
        #hx = self.last_timestep(encoder_hs.contiguous(), entext_len)
        hx = encoder_h_n.squeeze(0)

        #encoder_hs = torch.transpose(encoder_hs, 0, 1).contiguous()
        # 转置  zh_embedding: zh_maxLen * B * zh_dim
        zh_embedding = torch.transpose(zh_embedding, 0, 1)
        

        #我们要用这两个变量去存储输出的数据(是variable类型),所以这两个变量不应该是variable，
        #它们就是一个容器，容纳输出的variable变量。
        logits = [0 for i in range(zh_embedding.size(0))]
        predicts = [0 for i in range(zh_embedding.size(0))]
        for i in range(zh_embedding.size(0)):
            
            if is_eval:
                if i == 0:
                    inputs_x = zh_embedding[i]
                else:
                    inputs_x = self.zh_embedding(predicts[i-1])
            else:
                #判断是否使用上一次预测的结果作为下一次的输入
                #将这个与放在for循环外面则是整个句子预测一次，放在这里则是每个词都预测一次
                if random.random() < teacher_forcing_ratio:
                    use_teacher_forcing = True 
                else:
                    use_teacher_forcing = False
                #use_teacher_forcing = True
                if use_teacher_forcing or i == 0:
                    inputs_x = zh_embedding[i]
                    #print('i==%d used:'% i,inputs_x.size())
                else:
                    inputs_x = self.zh_embedding(predicts[i-1])
                    #print('i=%d unused:'% i,inputs_x.size())
   
            #hx: B * zh_hidden_size;  cx; B * zh_hidden_size
            hx, cx = self.lstm_cell(inputs_x,(hx, cx))
        
            #---------------- add attention-----------------------#
            if self.atten_mode != None:
                #atten_weight--> B * 1 * maxLen. it is 'at' in paper
                context = self.atten(encoder_hs, hx)
                #print(atten_weight)

                #context --> B * 1 * zh_hidden_size  it is 'ct' in paper
                #context = atten_weight.bmm(encoder_hs)

                #context --> B * zh_hidden_size
                #context = context.squeeze(1)

                #print('context size \n',context)
                #print('hx size \n', hx)
                hx_atten = self.ht_(torch.cat((context, hx), 1))
                hx_atten = F.tanh(hx_atten)
                
            #----------------end attention------------------------#
            #logits[i] 第i个decoder预测成每个词的概率（这里没有用softmax归一化）
            logits[i] = self.hx2zh_voc(hx_atten) 

            #选择概率最大的那个作为预测的结果
            _, predicts[i] = torch.max(logits[i], 1)
            
            logits[i] = logits[i].view(1, logits[i].size(0), logits[i].size(1))
            predicts[i] = predicts[i].view(1, predicts[i].size(0))

           
            
        logits = torch.cat(logits)
        predicts = torch.cat(predicts, 0)
        #logits --> zh_maxLen * B * zh_voc so change it to B* L * zh_voc
        #predicts --> zh_maxLen * B   so change it to B * L
        #predicts 转成data是为了在预测的时候可以使用
        return logits.transpose(0, 1), predicts.contiguous().transpose(0, 1).data.cpu()



