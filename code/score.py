import numpy as np
import random
import h5py
import sys, os
import nltk

from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu

def BLEUscore(zh_predicts, zh_answer, weight=(0.5,0.5,0,0)):
    """
    zh_predicts: 预测出来的句子，形式为['我来了','我来了', ... ] , B * Len
    zh_answer: 句子标签，形式为['我来了','我来了', ... ]
    """
    chencherry = SmoothingFunction()
    
    score = 0
    num = len(zh_predicts)
    for i in range(num):
        reference = []
        candidate = []
        for pre in zh_predicts[i]:
            candidate.append(pre)
        for ans in zh_answer[i]:
            reference.append(ans)
        #print(label, reference)
        
        if(len(candidate) != 0 and len(reference)!=0):
            score += sentence_bleu([reference], candidate,
                                   weights=weight, 
                                   smoothing_function = chencherry.method4)
        
    return score / num

