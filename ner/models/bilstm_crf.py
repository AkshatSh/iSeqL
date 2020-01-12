import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from .utils import log_sum_exp

try:
    import constants
except:
    from .. import constants

# to ensure replication of results is deterministic
# torch.manual_seed(1)

class BiLSTM_CRF(nn.Module):
    '''
    This model is a BiLSTM CRF for Named Entity Recognition, this involes a Bidirectional 
    LSTM to compute the features for an input sentence and then convert the computed features 
    to a tag sequence through a tag decoding CRF (Conditional Random Field)
    '''
    def __init__(self, vocab, tag_set, embedding_dim, hidden_dim, batch_size):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.tag_set = tag_set
        self.tag_set_size = len(tag_set)
        self.batch_size = batch_size

        # Embedder for the input sentence into the embedding dimension
        self.embedding = nn.Embedding(len(vocab), self.embedding_dim)

        # Bidirectional LSTM for computing the CRF features 
        # Note: there is an output vector for each direction of the LSTM
        #   hence the hidden dimension is // 2, so that the output of the LSTM 
        #   is size hidden dimension due to the concatenation of the forward and backward
        #   LSTM pass
        self.lstm = nn.LSTM(
            embedding_dim,
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
    
    def compute_lstm_features(self, sentence, mask):
        '''
        Given an input encoded sentence, compute the LSTM features
        for the CRF 

        Essentially run the Bidirectional LSTM and embedder
        '''
        device = sentence.device
        self.hidden = self.init_hidden(sentence.shape[0], device)
        # ('sentence', sentence.shape, len(sentence))
        # print('vocab', len(self.vocab))
        long_sentence = sentence.long()
        embeded_sentence = self.embedding(long_sentence) # .view(len(sentence), 64, -1)

        # embeded_sentence is now (batch_size x max_length x embedding dim)

        # print(embeded_sentence.shape)
        # print(self.hidden[0].shape)
        empty_elements = 0

        # embeded_sentence = nn.utils.rnn.pack_padded_sequence(
        #     embeded_sentence,
        #     mask.sum(1).int(),
        #     batch_first=True,
        # )
        lstm_output, self.hidden = self.lstm(embeded_sentence, self.hidden)

        # fix graph retention problem
        self.hidden = (
            torch.autograd.Variable(self.hidden[0].data, requires_grad=True), 
            torch.autograd.Variable(self.hidden[1].data, requires_grad=True),
        )
        # lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
        #     lstm_output,
        #     batch_first=True,
        # )

        # print(lstm_output.shape)
        # lstm_output = lstm_output.view(len(sentence), self.hidden_dim)

        # lstm output is now (batch_size x max_length x hidden_dim)

        features = self.tag_projection(lstm_output)

        # features is now (batch_size x max_length x tag_set size)
        return features
    
    def viterbi_decode(self, features, mask):
        '''
        A viterbi decoder for the features to compute the best 
        tag sequence 
        '''
        device = features.device
        back_pointers = torch.LongTensor().to(device)

        # create a score tensor of shape (BATCH_SIZE, tag_set_size)
        score = torch.Tensor(features.shape[0], len(self.tag_set)).fill_(-10000.).to(device)

        # cost is 0 to start with start
        score[:, self.tag_set(constants.START_TOKEN)] = 0. 

        for t in range(features.shape[1]):
            # go over all items in the sequence
            temp_bptr = torch.LongTensor().to(device)
            temp_score = torch.Tensor().to(device)

            for i in range(self.tag_set_size):
                # go over each possibleå tag
                # get the next max with the best transition score

                # m[0] is the max value m[1] is the max index for each transition
                m = [j.unsqueeze(1) for j in torch.max(score + self.transitions[i], 1)]
                temp_bptr = torch.cat((temp_bptr, m[1]), 1) # store the index of best previous
                temp_score = torch.cat((temp_score, m[0]), 1) # store the best score

            # add the bptrs 
            back_pointers = torch.cat((back_pointers, temp_bptr.unsqueeze(1)), 1)
            score = temp_score + features[:, t] # add on the emissions scores

        best_score, best_tag = torch.max(score, 1)

        # back track the decoding

        back_pointers = back_pointers.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(features.shape[0]):
            x = best_tag[b]

            # find out the length of the current sequence
            l = int(mask[b].sum().view(-1).data.tolist()[0])
            for temp_bptr in reversed(back_pointers[b][:l]):
                x = temp_bptr[x]
                best_path[b].append(x)
            best_path[b].pop()
            best_path[b].reverse()

        return best_path
    
    def forward(self, sentence, sentence_chars, s_ids: torch.Tensor = None):
        # compute LSTM features
        mask = sentence.data.gt(0).float()
        features = self.compute_lstm_features(sentence, mask)

        # compute the best path through viterbi
        # decoding
        tag_seq = self.viterbi_decode(features, mask)
        return tag_seq
    
    def compute_loss_viterbi(self, sentence, tagged):
        '''
        Given an input sequence and the correct taggging compute
        the difference in the viterbi scores of the output and 
        the computed output
        '''
        pass
    
    def compute_mle(self, sentence, sentence_chars, tagged, s_ids=None):
        '''
        To compute the MLE we want to solve

        $$ w* = \argmax{w} \sum_{n=1}^{N} w^t \phi (x_n, y_n) - \log(Z(x_n, w)) $$

        where Z is the partition function and w^t\phi computes the sentence score of 
        the input
        '''
        
        # mask will be 1 for all items that are not padding, and 0 for padding
        mask = sentence.data.gt(0).float()

        # features are the outputs from the LSTM encoders
        features = self.compute_lstm_features(sentence, mask)

        # compute the partitions and gold score of the features to compute
        # the MLE loss
        partition = self.compute_partion(features, mask)
        # print('fwd',torch.mean(partition))
        # gold_score = self.score_sentence(features, tagged, mask)
        gold_score = self.score(features, tagged, mask)
        # print('gold', torch.mean(gold_score))

        # to make this a gradient descent problem we want to minimze, hence
        # the order is switched from the MLE equation
        return partition - gold_score
    
    def compute_partion(self, features, mask):
        '''
        This is the log sum of all feature vectors invovled in computing
        the tag sequence of the input sentence
        '''

        # score (batch_size x max_length x tag_set_size)
        #print(features.shape)
        #print((features.shape[0], len(self.tag_set)))
        device = features.device
        score = torch.Tensor(features.shape[0], len(self.tag_set)).fill_(-100000).to(device)
        # print(score.shape)
        score[:, self.tag_set(constants.START_TOKEN)] = 0
        trans = self.transitions.unsqueeze(0) # (1, tags_size, tags_size)

        for time_step in range(features.shape[1]): 
            # iterate over all the time steps in the sequence
            mask_t = mask[:, time_step].unsqueeze(1) # (batch_size, 1)
            emission = features[:, time_step].unsqueeze(2) # (batch_size, tags_size, 1)

            # score is the sum of the previous scores along with the computation of
            # emission and transitions scores, where emission is based on the features
            score_t = score.unsqueeze(1) + emission + trans # (batch_size, tags_size, tags_size)

            # we sum using log_sum_exp to work in the log space and reduce the magnitude of
            # the numbers
            score_t = log_sum_exp(score.unsqueeze(1) + emission + trans) # (batch_size, tags_size)

            # include the current score but only for the batches that aren't filtered by the mask
            # add to the previous score of everything that has been filtered
            score = score_t * mask_t + score * (1 - mask_t)

        score = log_sum_exp(score)
        return score

    def compute_uncertainty(
        self,
        sentence: torch.Tensor,
        characters: torch.Tensor,
        s_ids: torch.Tensor = None,
    ) -> float:
        best_seq = self.forward(sentence, characters)
        score = self.compute_mle(
            sentence,
            characters, 
            torch.LongTensor(best_seq),
        )
        return score

    # def explained_score_sentence(self, features, tagged, mask):
    #     
    #     Compute the score of the correct tagged sentence for evaluation

    #     features: (batch_size, max_sentence, tags_size)
    #     predictions: (batch_size, max_sentence)
    #     '''
    #     score = torch.Tensor(features.shape[0]) # a loss score for each batch

    #     features = features.unsqueeze(3) # (batch_size, max_sentence, tags_size, 1)

        # start = torch.Tensor(tagged.shape[0], 1).fill_(self.tag_set(constants.START_TOKEN)).long().to(device)

    #     start = torch.Tensor(tagged.shape[0], 1).fill_(self.tag_set(constants.START_TOKEN)).long()

        # for t in range(features.shape[1]):
        #     # iterate over each term in the sequence
        #     mask_t = mask[:, t] # get the mast for the current time step

    #     for t in range(features.shape[1]):
    #         # iterate over each term in the sequence
    #         mask_t = mask[:, t] # get the mast for the current time step

    #         # emissions:
    #         #   compute the emission to the next predicted tag concatentation for all 
    #         #   the batches
    #         emissions = torch.cat([features[b, t, tagged[b, t + 1]] for b in range(features.shape[0])])

    #         # transitions:
    #         #   compute the concatentation of all the transition scores for the current word to the next
    #         #   for each batch
    #         transitions_t = torch.cat([transitions[sequence[t + 1], sequence[t]] for sequence in tagged])

    #         # update the score to add the sum of the emissions and transitions
    #         score += (emissions + transitions_t) * mask_t
        
    #     return score
    
    def score(self, h, y, mask): # calculate the score of a given sequence
        device = y.device
        BATCH_SIZE = y.shape[0]
        score = torch.Tensor(BATCH_SIZE).fill_(0.).to(device)
        h = h.unsqueeze(3)
        start = torch.Tensor(y.shape[0], 1).fill_(self.tag_set(constants.START_TOKEN)).long().to(device)
        y = torch.cat([start, y.long()], dim=1)
        trans = self.transitions.unsqueeze(2)
        for t in range(h.size(1)): # iterate through the sequence
            mask_t = mask[:, t]
            emit = torch.cat([h[b, t, y[b, t + 1]] for b in range(BATCH_SIZE)])
            trans_t = torch.cat([trans[seq[t + 1], seq[t]] for seq in y])
            score += (emit + trans_t) * mask_t
        return score