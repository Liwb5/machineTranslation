import dataProcess as dp 

class Transformer:
    def __init__(self, inputlang, outputlang):
        self.inputlang = inputlang
        self.outputlang = outputlang



    def index2text(self, index, lang):
        assert lang == 'en' or lang == 'zh'

        text = ''
        if lang == 'en':
            for i in range(len(index)):
                token = self.inputlang.index2word(index[i])
                text += token

        else:
            for i in range(len(index)):
                text += self.outputlang.index2word(index[i])

        return text