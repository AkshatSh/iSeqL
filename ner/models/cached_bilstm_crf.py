import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from allennlp.modules.elmo import Elmo, batch_to_ids

from .crf import CRF
from .utils import log_sum_exp
from .elmo import FrozenELMo

from ner import constants
from ner import vocab
from ner.models import cached_embedder

class CachedBiLSTMCRF(CRF):
    '''
    This model is a BiLSTM CRF for Named Entity Recognition, this involes a Bidirectional 
    LSTM to compute the features for an input sentence and then convert the computed features 
    to a tag sequence through a tag decoding CRF (Conditional Random Field)
    '''
    def __init__(
        self,
        vocab: vocab.Vocab,
        tag_set: vocab.Vocab,
        hidden_dim: int,
        batch_size: int,
        embedder: cached_embedder.CachedEmbedder,
    ):
        super(CachedBiLSTMCRF, self).__init__(
            vocab,
            tag_set,
            embedder.dimensions(),
            hidden_dim,
            batch_size,
        )
        self.embedder = embedder
        self.embedding_dim = embedder.dimensions()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.tag_set = tag_set
        self.tag_set_size = len(tag_set)
        self.batch_size = batch_size

        # Bidirectional LSTM for computing the CRF features 
        # Note: there is an output vector for each direction of the LSTM
        #   hence the hidden dimension is // 2, so that the output of the LSTM 
        #   is size hidden dimension due to the concatenation of the forward and backward
        #   LSTM pass
        self.lstm = nn.LSTM(
            self.embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Project LSTM outputs to the taget set space
        self.tag_projection = nn.Linear(hidden_dim, len(self.tag_set))

        self.hidden = self.init_hidden(batch_size)
    
    def can_cache(self):
        return True
    
    def init_hidden(self, batch_size, device='cpu'):
        '''
        Initialize the hidden dimensions of the LSTM outputs
        '''
        return (
            torch.randn(2, batch_size, self.hidden_dim // 2).to(device),
            torch.randn(2, batch_size, self.hidden_dim // 2).to(device),
        )
    
    def _compute_embedded_features(self, embeded_sentence) -> torch.Tensor:
        self.hidden = self.init_hidden(embeded_sentence.shape[0], embeded_sentence.device)
        lstm_output, self.hidden = self.lstm(embeded_sentence, self.hidden)

        # fix graph retention problem
        self.hidden = (
            torch.autograd.Variable(self.hidden[0].data, requires_grad=True), 
            torch.autograd.Variable(self.hidden[1].data, requires_grad=True),
        )

        # lstm output is now (batch_size x max_length x hidden_dim)

        features = self.tag_projection(lstm_output)

        # features is now (batch_size x max_length x tag_set size)
        return features
    
    def compute_lstm_features(self, sentence, sentence_chars, mask) -> torch.Tensor:
        embeded_sentence = self.embedder(sentence_chars)
        return self._compute_embedded_features(embeded_sentence)
    
    def compute_cached_lstm_features(self, s_ids, sentence, sentence_chars, mask) -> torch.Tensor:
        '''
        Given an input encoded sentence, compute the LSTM features
        for the CRF 

        Essentially run the Bidirectional LSTM and embedder
        '''
        embeded_sentence = self.embedder.batched_forward_cached(s_ids, sentence)
        # embeded_sentence is now (batch_size x max_length x embedding dim)
        return self._compute_embedded_features(embeded_sentence)
