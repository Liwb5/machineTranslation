

class Transformer(object):
    def __init__(self, inputLang, outputLang, params):
        """
        brief: 导入机器翻译的源语言词典与目标语言词典，这样才能通过下标索引到对应的词。
            
        """
        self.inputLang = inputLang
        self.outputLang = outputLang
        self.params = {v: k for k, v in params.items()}
        
    def index2text(self, index, lang):
        """
        brief: 将句子下标转换成相应的句子。
        params:
            index: 句子下标
            lang: 表明哪种语言。eng==英语，fra==法语
        """
        
        sentence = ''
        if lang == self.inputLang.name:
            for i in range(len(index)):
                if index[i] == self.params['EOS_token']:
                    break
                if index[i] == self.params['SOS_token']:
                    continue
                    
                word = self.inputLang.index2word[index[i]] + ' '
                sentence += word
                
        elif lang == self.outputLang.name:
            for i in range(len(index)):
                if index[i] == self.params['PAD_token'] or index[i] == self.params['EOS_token']:
                    break
                if index[i] == self.params['SOS_token']:
                    continue
                    
                word = self.outputLang.index2word[index[i]]+ ' '
                sentence += word 
        
        return sentence
            
        