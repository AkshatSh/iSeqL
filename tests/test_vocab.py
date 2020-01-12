import unittest

import torch

from ner.vocab import (
    Vocab,
    build_vocab,
    build_output_vocab,
)

from ner import constants
from . import utils

SAMPLE_WORDS = ['Here', 'is', 'a', 'series', 'of', 'words', 'with', 'words', 'repeated']
TAGS = ['B-word', 'B-random', 'I-random', 'O', 'O', 'B-random', 'I-random']

UNIQUE_WORDS = set(SAMPLE_WORDS)
UNIQUE_TAGS = set(TAGS)

class TestVocabulary(unittest.TestCase):

    def contains(self, vocab: Vocab, word: str):
        '''
        check if the vocab object has the word argument
        '''
        unk_index = vocab(constants.UNKNOWN_TOKEN)
        assert not vocab(word) == unk_index

    def _has_words(self, vocab: Vocab, words: list) -> None:
        '''
        check if the vocab object has all the words in the words
        argument
        '''
        for word in words:
            self.contains(vocab, word)

    def test_unk(self) -> None:
        '''
        Test to make sure the vocab object supports unking
        '''
        vocab = build_vocab(SAMPLE_WORDS)

        unk_index = vocab(constants.UNKNOWN_TOKEN)
        
        # encode
        assert vocab('random_word_here') == unk_index

        # decode
        assert vocab.get_word(unk_index) == constants.UNKNOWN_TOKEN 
        

    def test_vocab_creation(self) -> None:
        '''
        make sure vocab creation is able to create vocabulary objects
        with the correct words
        '''
        vocab = build_vocab(SAMPLE_WORDS)
        self._has_words(vocab, UNIQUE_WORDS)
        for token in constants.SPECIAL_TOKENS:
            if token == constants.UNKNOWN_TOKEN:
                continue
            self.contains(vocab, token)
    
    def test_tag_vocab_creation(self) -> None:
        '''
        make sure vocab objects are able to create tag vocabularies
        with the right words

        in particular, tag vocabularies don't contain UNK tokens
        '''
        vocab = build_output_vocab(TAGS)
        for tag in UNIQUE_TAGS:
            vocab(tag)

        for token in constants.SPECIAL_TOKENS:
            if token == constants.UNKNOWN_TOKEN:
                continue
            
            # will throw an exception if not there

            vocab(token)
    
    def test_vocab_decode(self) -> None:
        '''
        test ability to decode a random string
        '''
        vocab = utils.build_sample_vocab()
        encoded_sentence = [i for i in range(len(vocab))]
        vocab.decode(encoded_sentence)

class TestBinaryVocabulary(unittest.TestCase):

    def contains(self, vocab: Vocab, word: str):
        '''
        check if the vocab object has the word argument
        '''
        o_token = vocab('O')
        assert not vocab(word) == o_token

    def _has_words(self, vocab: Vocab, words: list) -> None:
        '''
        check if the vocab object has all the words in the words
        argument
        '''
        for word in words:
            self.contains(vocab, word)
    
    def test_contains(self):
        binary_vocab = utils.build_sample_binary_vocab()
        self._has_words(binary_vocab, ['B-tag', 'I-tag'])

    def test_generic(self) -> None:
        '''
        Test various attributes of binary vocab
        '''
        binary_vocab = utils.build_sample_binary_vocab()
        vocab = utils.build_sample_tag_vocab()
        assert len(binary_vocab) == len(vocab)
    
    def test_vocab_decode(self) -> None:
        '''
        test ability to decode a random string
        '''
        vocab = utils.build_sample_binary_vocab()
        encoded_sentence = [i for i in range(len(vocab))]

        # since <PAD> is a break token, this ensures everything in the
        # vocab gets decoded
        vocab.decode(torch.Tensor(encoded_sentence))
        encoded_sentence.reverse()
        vocab.decode(torch.Tensor(encoded_sentence))