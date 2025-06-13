from collections import defaultdict, counter
import regex as re

class bpetokenizer:
    def __init__(self,vocab_size=1000):
        self.vocab_size
        self.bpe_codes ={}
        self.bpe_vocab = {}

    #makes vocabulary
    def get_vocab(self,text_list):
        vocab = counter()
        for s in text_list:
            chars  = list(s) + "</w>"
            vocab[chars] +=1
        return vocab
    
    #make the [char,next char] dictionarty with its frequency. returning pairs with their frequency
    def get_stat(self,vocab):
        pairs = defaultdict(int)
        for word,freq in vocab.items:
            for i in range(len(word)-1):
                pairs[(word[i],word[i+1])] +=1
        return pairs
    
    #merges the letters and replaces it in the vocab, and there u go u got subwords. 
    def merge(self,vocab,pairs):
        new_vocab = {}
        bigram = re.escape(' '.join(pairs))
        p =re.compile(r'(?<!\S)' + bigram + r'(?<!\S)')

        for words in vocab:
            words_str = " ".join(words)
            word_str = p.sub(''.join(pairs),word_str)
            new_vocab[tuple(words_str)] = vocab[words]
        
        return new_vocab

    def train(self,texts):
        words =[]
        for sentence in texts.list():
            for word in sentence:
                words.append(word)

        vocab = self.get_vocab(words)
        for i in range(self.vocab_size):
            pairs = self.get_stat(vocab)
            if not pairs:
                break
            best = max(pairs,key = pairs.get)
            vocab = self.merge(best,vocab)
            self.bpe_codes[best] = i  # Keep track of merge order

        self.bpe_vocab = set()
        for word in vocab:
            for sub in word:
                self.bpe_vocab.add(sub)

    def apply_bpe(self, word):
        word = list(word) + ['</w>']
        while True:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            mergeable = {pair: self.bpe_codes.get(pair, float('inf')) for pair in pairs}
            if not mergeable:
                break
            best = min(mergeable.items(), key=lambda x: x[1])[0]
            if best not in self.bpe_codes:
                break
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word)-1 and (word[i], word[i+1]) == best:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word


            
