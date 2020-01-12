import nltk
import pickle
import argparse
import tqdm

try:
    # this is run in the package case
    from .constants import (
        UNKNOWN_TOKEN,
        PAD_TOKEN,
        START_TOKEN,
        END_TOKEN,
    )
except:
    # this is run in the script case
    from constants import (
        UNKNOWN_TOKEN,
        PAD_TOKEN,
        START_TOKEN,
        END_TOKEN,
    )

class Vocab(object):
    '''
    Vocabulary object, supports two functions:
        vocab.add_word(word): creates a new entry adding that word to the dictionary
        vocab(word): returns the index representing the word
    '''
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = {}
        self.idx = 0
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.word_count[word] = 1
            self.idx += 1
        else:
            self.word_count[word] += 1
    
    def get_word(self, index):
        return self.idx2word[index] if index in self.idx2word else None
    
    def get_all(self):
        return self.word2idx.keys()
    
    def __call__(self, word):
        return self.word2idx[word] if word in self.word2idx else self.word2idx[UNKNOWN_TOKEN]

    def __len__(self):
        return len(self.word2idx)
    
    def decode(self, encoding):
        curr_sentence = []
        for j in range(len(encoding)):
            if encoding[j] == self.word2idx[PAD_TOKEN]:
                break
            word = self.get_word(int(encoding[j].item()))
            curr_sentence.append(word)
        
        return curr_sentence

class BinaryVocab(object):
    '''
    Vocabulary object to convert any classification problem from multiclass to single class
    '''
    def __init__(
        self,
        vocab: Vocab,
        select_class: str,
        default_value: str = 'O'
    ):
        self.valid_words = [START_TOKEN, END_TOKEN, PAD_TOKEN, 'O']
        self.valid_indexes = [vocab(word) for word in self.valid_words]
        self.vocab = vocab
        for word in vocab.word2idx:
            if len(word) > 3 and word[2:] == select_class:
                self.valid_words.append(word)
                self.valid_indexes.append(vocab(word))
    
    def get_word(self, index: int):
        return self.vocab.get_word(index)
    
    def get_all(self):
        return self.valid_words
    
    def __call__(self, word):
        if word in self.valid_words:
            return self.vocab(word)
        else:
            return self.vocab('O')

    def __len__(self):
        return len(self.vocab)
    
    def decode(self, encoding):
        curr_sentence = []
        for j in range(len(encoding)):
            if encoding[j] == self.vocab.word2idx[PAD_TOKEN]:
                break
            word = self.get_word(int(encoding[j].item()))
            curr_sentence.append(word)
        
        return curr_sentence

def unk_vocab(vocab: Vocab, unk_threshold: int = 2) -> Vocab:
    words = []
    for word, count in vocab.word_count.items():
        if count >= unk_threshold:
            words.append(word)
    return build_vocab(words)

def build_vocab(words):
    '''
    Given a set of words constructs and returns a vocabulary object
    '''
    vocab = Vocab()
    vocab.add_word(PAD_TOKEN)
    for word in tqdm.tqdm(words):
        vocab.add_word(word)
    
    # Special tokens
    vocab.add_word(START_TOKEN)
    vocab.add_word(END_TOKEN)
    vocab.add_word(UNKNOWN_TOKEN)

    return vocab

def build_output_vocab(words):
    vocab = Vocab()
    vocab.add_word(PAD_TOKEN)
    for word in tqdm.tqdm(words):
        vocab.add_word(word)
    
    # special tokens
    vocab.add_word(START_TOKEN)
    vocab.add_word(END_TOKEN)
    
    return vocab
