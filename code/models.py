# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import nltk

#--------some hyperparameters-------------------#
use_cuda = torch.cuda.is_available()

SOS_token = 0
EOS_token = 1

teacher_forcing_ratio = 0.9  #在训练时解码器使用labels（平行预料）进行训练的概率
MAX_LENGTH = 50
LEARNING_RATE = 0.01
HIDDEN_STATE = 256
N_LAYERS = 1   #对一个句子循环RNN训练的次数
DROPOUT_P = 0.1
modelName = 'first'


from hyperboard import Agent
agent = Agent(address='127.0.0.1',port=5100)
#agent = Agent(address='172.18.216.69',port=5000)
hyperparameters = {'learning rate':LEARNING_RATE,
                   'max_length':MAX_LENGTH,
                  'teacher_forcing_ratio':teacher_forcing_ratio,
                  'hidden_state':HIDDEN_STATE,
                   'n_layers':N_LAYERS
                  }
name = agent.register(hyperparameters, 'loss',overwrite=True)


class EncoderRNN(nn.Module):
    #input_size是指词典的大小(毕竟要建立embedding)，hidden_size是hidden_state的维度
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    #input是一个句子(这个句子已经通过数据处理的类转换成下标，这样可以对应一个embedded)
    #hidden 是上一个迭代中的hidden，即pre_hidden
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        #这个n_layers==1其实就是只相当于一个cell，对一个input(单词)和上一个hidden state
        #这里做了一个gru操作。n_layers大于1则是对同一个东西迭代多次，也许效果会好。
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
        
#hidden_size都是说hidden_state的维度，要和encoder一致。
#output_size是目标语言的词典大小，因为输出的是所有词的概率
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
        
#hidden_size都是说hidden_state的维度，要和encoder一致。
#output_size是目标语言的词典大小，因为输出的是所有词的概率
#max_length是句子的最大长度(之前被限制了，以后看看能否不要这个限制)
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    #input是一个目标句子中的某个词(这个词已经通过数据处理的类转换成下标，这样可以对应一个embedded)。
    #当然，在进行预测的时候就不会是输入目标句子的词了。而是它预测出来的词
    #hidden 是上一个迭代中的hidden，即pre_hidden
    #encoder_output是encoder最后一个输出，不过在这里它并没有被使用到
    #encoder_outputs是encoder每次输出(y1,y2,...,yn)的组成tensor，格式跟input一样，只不过它是句子，而不是某个词。
    def forward(self, input, hidden, encoder_output, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))#每个词的概率
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result



#########################training#####################################
### 将数据进行处理成pytorch变量，方便转换成embedded
#返回词对应的下标
def indexesFromSentence(lang, sentence):
    #return [lang.word2index[word] for word in sentence.split(' ')]
    result = []
    all_lang_keys = lang.word2index.keys()
    for word in sentence.split(' '):
        if word in all_lang_keys:#判断词是否在词典中，因为词典中有些出现次数太少的词被删掉了
            result.append(lang.word2index[word])
    return result

#将词转换成variable
def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)#一个句子最后都要加上一个结束符
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result

    
def variablesFromPair(pair,input_lang, output_lang):
    #注意这里要先解码，因为保存到h5py里面的时候要编码，所以现在要解码
    input_variable = variableFromSentence(input_lang, pair[0].decode('utf-8'))
    target_variable = variableFromSentence(output_lang, pair[1].decode('gb2312'))
    return (input_variable, target_variable)



def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    #input_variable已经在函数外变成了tensor了，tensor的元素是词的下标
    input_length = input_variable.size()[0]#source sentence 的长度
    target_length = target_variable.size()[0]#目标句子的长度

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))#max_length=10，也就是句子的最长长度，hidden_size是256，所以encoder_outputs是矩阵10X256
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):#句子有多长就迭代多少次
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]#将每个词encoder的output记录下来

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden#encoder最后一层的hidden_state传给decoder作为decoder的第一个hidden_state

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
        	#encoder_outputs作为decoder的输入，是为了改变attention。
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])#这两个变量是什么形式
            decoder_input = target_variable[di]  # Teacher forcing这个是直接给答案，也就是一个单词，进入decoder里面再变成词向量

    else:#这边是不直接给答案，而是每次output那里选择概率最大的作为下一个输入
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            #从decoder的类中可以知道，decoder_output是softmax出来的，即所有词的概率。
            #topk函数是查找最大的K 个数，这里参数是1，topv就是value，topi是index，也就是词对应的下标
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()#如何理解这一步反向梯度对encoder和decoder都有效

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent) #总时间
    rs = es - s        #总时间减去已经运行的时间等于还剩下的时间
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def trainIters(encoder, decoder, inputlang, outputlang, pairs, n_iters, print_every=1000, plot_every=100, learning_rate=0.01, save_model_every=10000):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    #考虑改成其他的优化器
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    #这里的training_pairs经过variableFromPair处理后，每个元素已经是一个tensor了，并且是单词所在的下标，为了可以和embedd匹配。
    #training_pairs = [variablesFromPair(random.choice(pairs))
    #                 for i in range(n_iters)]
    #print(random.choice(training_pairs)[0].data)
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        #training_pair = training_pairs[iter - 1]
        #################@#%…………&&&
        
        training_pair = variablesFromPair(random.choice(pairs),inputlang,outputlang)
        input_variable = training_pair[0]
        target_variable = training_pair[1]
        #如果句子太长，丢弃它
        if len(input_variable) > MAX_LENGTH or len(target_variable) > MAX_LENGTH:
            pass
        else:     
            #print(training_pair[0])
            loss = train(input_variable, target_variable, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            agent.append(name, iter, plot_loss_avg)
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            
        if iter % save_model_every == 0:
            torch.save(encoder,'../models/encoder{0}.m{1}'.format(modelName,iter))
            torch.save(decoder,'../models/decoder{0}.m{1}'.format(modelName,iter))
            pass

        
        
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

######################################################################
# Evaluation
# ==========
#
# Evaluation is mostly the same as training, but there are no targets so
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there. We also store the decoder's
# attention outputs for display later.
#

def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):#注意这里跟训练的时候不一样，训练的时候用的是target_length。这里因为要输出句子，而代码限定了句子的最大长度。
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            #decoded_words.append('<EOS>')  #检测到结束符就停止
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
#

def evaluateRandomly(encoder, decoder, inputlang, outputlang,pairs, n=100):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0].decode('utf-8'))
        print('=', pair[1].decode('gb2312'))
        output_words, attentions = evaluate(encoder, decoder, pair[0].decode('utf-8'),inputlang, outputlang, max_length=100)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
       
        
def calculateValidData_BLEU_Score(encoder, decoder, inputlang, outputlang, pairs):
    predict_words = []
    bleu_score = 0.0
    for pair in pairs:
        output_words, attentions = evaluate(encoder, decoder, pair[0].decode('utf-8'),inputlang, outputlang, max_length=100)
        #output_sentence = ' '.join(output_words)
        label_words = nltk.word_tokenize(pair[1].decode('gb2312'))
        bleu_score += nltk.translate.bleu_score.sentence_bleu([label_words],output_words)
    bleu_score = bleu_score/len(pairs)
    print('bleu_score is:',bleu_score)
    return bleu_score
        
######for test data__________________
def evaluateForTestData(encoder, decoder, inputlang, pairs, n=100):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0].decode('utf-8'))
        output_words, attentions = evaluate(encoder, decoder, pair[0].decode('utf-8'),inputlang, outputlang, max_length=100)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
        
def predictTestData(encoder, decoder, inputlang, pairs):
    predicted_sentences = []
    for pair in pairs:
        output_words, attentions = evaluate(encoder, decoder, pair.decode('utf-8'),inputlang, outputlang, max_length=100)#需要知道pair是否需要下标
        output_sentence = ' '.join(output_words)
        predicted_sentences.append(output_sentence)
    return predicted_sentences
        
        
if __name__=='__main__':
    hidden_size = 256
    encoder1 = EncoderRNN(inputlang.n_words, hidden_size)
    attn_decoder1 = AttnDecoderRNN(hidden_size, outputlang.n_words,
                                   1, dropout_p=0.1)

    if use_cuda:
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()

    trainIters(encoder1, attn_decoder1, inputlang, outputlang, 80000, print_every=100, save_model_every=2000)

