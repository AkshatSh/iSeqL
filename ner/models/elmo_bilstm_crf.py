import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from allennlp.modules.elmo import Elmo, batch_to_ids

from .crf import CRF
from .utils import log_sum_exp
from .elmo import FrozenELMo

from ner import constants


def load_elmo():
    '''
    load a pretrained elmo model
    '''
    with torch.no_grad():
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        return Elmo(options_file, weight_file, 2, dropout=0)

# to ensure replication of results is deterministic
# torch.manual_seed(1)

class ELMo_BiLSTM_CRF(CRF):
    '''
    This model is a BiLSTM CRF for Named Entity Recognition, this involes a Bidirectional 
    LSTM to compute the features for an input sentence and then convert the computed features 
    to a tag sequence through a tag decoding CRF (Conditional Random Field)
    '''
    def __init__(self, vocab, tag_set, hidden_dim, batch_size, freeze_elmo: bool=True):
        super(ELMo_BiLSTM_CRF, self).__init__(vocab, tag_set, 1024, hidden_dim, batch_size)
        self.embedding_dim = 1024 # elmo embedding size
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.tag_set = tag_set
        self.tag_set_size = len(tag_set)
        self.batch_size = batch_size
        self.elmo = FrozenELMo.instance() if freeze_elmo else load_elmo()

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


        # Matrix of transition scores, transitions[i][j] is the cost to transition
        # to tag[i] from tag[j]
        self.transitions = nn.Parameter(
            torch.randn(len(self.tag_set), len(self.tag_set))
        )

        # very high cost to transition to the start token
        self.transitions.data[self.tag_set(constants.START_TOKEN), :] = -100000

        # very high cost to transition from the end token
        self.transitions.data[:, self.tag_set(constants.END_TOKEN)] = -100000

        # very high cost to transition from pading to anything aside from padding 
        self.transitions.data[:, self.tag_set(constants.PAD_TOKEN)] = -100000

        # very high cost to transition from anything to padding aside from EOS
        self.transitions.data[self.tag_set(constants.PAD_TOKEN), :] = -100000

        # special cases of earlier
        self.transitions.data[
            self.tag_set(constants.PAD_TOKEN), 
            self.tag_set(constants.PAD_TOKEN)
        ] = 0

        self.transitions.data[
            self.tag_set(constants.PAD_TOKEN), 
            self.tag_set(constants.END_TOKEN)
        ] = 0

        self.hidden = self.init_hidden(batch_size, 'cpu')
    
    def init_hidden(self, batch_size, device):
        '''
        Initialize the hidden dimensions of the LSTM outputs
        '''
        return (
            torch.randn(2, batch_size, self.hidden_dim // 2).to(device),
            torch.randn(2, batch_size, self.hidden_dim // 2).to(device),
        )
    
    def compute_lstm_features(self, sentence, sentence_chars, mask):
        '''
        Given an input encoded sentence, compute the LSTM features
        for the CRF 

        Essentially run the Bidirectional LSTM and embedder
        '''
        device = sentence.device
        self.hidden = self.init_hidden(sentence.shape[0], device)
        long_sentence = sentence.long()

        raw_sentence = []
        if sentence_chars is not None:
            character_ids = sentence_chars
        else:
            for i in range(len(sentence)):
                curr_sentence = []
                for j in range(len(sentence[i])):
                    if sentence[i][j] == 0:
                        continue
                    word = self.vocab.get_word(int(sentence[i][j].item()))
                    curr_sentence.append(word)
                raw_sentence.append(curr_sentence)
            character_ids = batch_to_ids(raw_sentence)
        embeddings = self.elmo(character_ids)
        embeded_sentence = embeddings['elmo_representations'][0]

        # embeded_sentence is now (batch_size x max_length x embedding dim)

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