import pickle

class Lang(object):
    def __init__(self, name, specialTokens={0: "_SOS_", 1: "_EOS_",2:"_PAD_"}):
        self.name = name
        self.word2count = {}  #记录词频
        self.word2index = {}  #记录词对应的下标
        self.index2word = specialTokens  #记录下标对应的词
        self.n_words = len(specialTokens)  #词的数量
        
    def countWordFreq(self, word):
        """
        统计词频，这样做是为了可以去掉一些词频比较低的词。不知道这种做法好不好。
        """
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] +=1

    def addSentence(self, sentence):
        """
        不断输入sentence（字符串的格式），构建词与下标的对应（词典），方便制作one-hot。
        
        params:
            sentence: string; i.e: 'I am a student.'
        """
        for word in sentence.split(' '):
            self.addWord(word)
            
            
    def addWord(self, word):
        """
        记录词的下标，统计词频。
        params:
            word: string; i.e: 'student'
        """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1
            
            
    def save(self, path):
        with open(path,'wb') as f:
            pickle.dump([self.name,self.word2index, self.word2count, self.index2word, self.n_words],f)
    
    def load(self,path):
        with open(path,'rb') as f:
            name, self.word2index, self.word2count, self.index2word, self.n_words = pickle.load(f)
        if self.name != name:
            print('error: Name error------------------------------!')
            
            
            
    def indexesFromSentence(self, sentence):
        indexes = []
        all_lang_keys = self.word2index.keys()
        for word in sentence.split(' '):
            if word in all_lang_keys:
                indexes.append(self.word2index[word])
        indexes.append(EOS_token)
        return indexes



























