import numpy as np
import random
import h5py
import sys, os
import nltk


from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu

def BLEUscore(tar_answers, tar_predicts, weight=(0.5,0.5,0,0)):
    """
    tar_answer: 句子标签，形式为['how are you','how are you ', ... ]
    tar_preidct: 预测出来的句子
    """
    chencherry = SmoothingFunction()
    
    score = 0
    num = len(tar_predicts)
    
    for i in range(num):
        reference = tar_answers[i].split(' ')
        candidate = tar_predicts[i].split(' ')
        #print(reference, candidate)
        #print(len(reference))
        if (len(reference)) != 0 and len(candidate) != 0:
            score += sentence_bleu([reference], candidate)#, 
                                   #smoothing_function=chencherry.method4)#bleu-4
            
    return score/num








#print(BLEUscore(['I am chinese . '], ['I am chinese so . ']))
#print('asdfj')



































