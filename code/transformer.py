import dataProcess as dp 

SOS_token = 0
EOS_token = 1
__PADDING__ = 2

class Transformer:
    def __init__(self, inputlang, outputlang):
        self.inputlang = inputlang
        self.outputlang = outputlang



    def index2text(self, indexes, lang):
        assert lang == 'en' or lang == 'zh'

        text = ''
        if lang == 'en':
            for i in range(len(indexes)):
                if indexes[i] == EOS_token:
                    break
                token = self.inputlang.index2word[indexes[i]]+ ' '
                text += token

        else:
            
            for i in range(len(indexes)):
                #print(indexes[i])
                if indexes[i] == EOS_token or indexes[i] == __PADDING__:
                    break
                token = self.outputlang.index2word[indexes[i]]
                text += token
        return text