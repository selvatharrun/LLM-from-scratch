import defaultdict

class bpetokenizer:
    def __init__(self,vocab_size):
        self.vocab_size = 1000
        self.bpe_codes ={}
        self.bpe_vocab = {}

    def get_vocab(self,text_list):
        vocab = {}
        for s in text_list:
            chars  = list(s) + "</w>"
            vocab[chars] +=1
        return vocab
    
    def 
