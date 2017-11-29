import numpy as np
import random
import h5py
import sys, os
import nltk

from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu

def BLEUscore(zh_predicts, zh_answer):
    """
    zh_predicts: 预测出来的句子，形式为[['我'，'来','了'],['我'，'来','了'] ... ]
    zh_answer: 句子标签，形式为[['我'，'来','了'],['我'，'来','了'] ... ]
    """
    chencherry = SmoothingFunction()
    
    score = 0
    num = len(zh_predicts)
    for i in range(num):
        reference = []
        label = []
        for pre in zh_predicts[i]:
            reference.append(pre)
        for ans in zh_answer[i]:
            label.append(ans)
        
        score += sentence_bleu([reference], label, smoothing_function = chencherry.method4)

    return score / num

