from typing import (
    List,
    Tuple,
    Dict,
)

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

try:
    import constants
except:
    from .. import constants

from ner import utils

USE_SMART = True

def get_ents(
    example: List[str],
    seq_label: List[str],
) -> Tuple[List[Tuple[int, int]], List[str]]:
    '''
    Convert a label to a list of word ranges and entities
    entities[i] = str, the entity corresponding to word_range[i]
    '''
    entities: list = []
    range_start : int = None
    seq_label = [] if seq_label is None else seq_label
    seq_class = None
    for i, label in enumerate(seq_label):
        if (label == 'O' or i == len(seq_label) - 1) and range_start is not None:
            entities.append((seq_class, example[range_start : i]))
            range_start = None
        elif label.startswith('B'):
            if range_start is not None:
                entities.append((seq_class, example[range_start : i]))
            seq_class = label[2:]
            range_start = i
    return entities

class DictionaryModel(nn.Module):
    '''
    This model gets a training example and sets the word in the training example to belong
    to the class passed in. In particular given a sentence x, and tagged sentence y all the
    words in x increment the counter of y by 1

    Inference involves selecting the maximum class from the input

    Simplistic model
    '''
    def __init__(
        self,
        vocab,
        tags,
        smart=False
    ):
        super(DictionaryModel, self).__init__()
        self.smart = smart
        self.vocab = vocab
        self.tags = tags
        self.classifier = Variable(torch.Tensor(
                len(vocab), 
                len(tags)
            ),
            requires_grad=False,
        )

        # list of class, ent
        self.dictionary: List[Tuple[str, str]] = []
    
    def check_contains(self, ent: str, sent: List[str]) -> Tuple[int, int]:
        i = 0
        while i < len(sent):
            if ent[0] == sent[i] and (i + len(ent)) < len(sent):
                # if we can go till the end of the sentence
                # check is substring
                found = True
                for j in range(len(ent)):
                    if ent[j] != sent[i + j]:
                        found = False
                
                if found:
                    return [i, i + len(ent)]
            i += 1
        
        return None
                
    
    def smart_forward_single(self, x: torch.Tensor) -> torch.Tensor:
        sent = self.vocab.decode(x.cpu().long())
        res = ['O'] * len(sent)
        for classifier_class, ent in self.dictionary:
            index_tuple = self.check_contains(ent, sent)
            if index_tuple is not None:
                start, end = index_tuple
                res[start] = f'B-{classifier_class}'
                for i in range(start + 1, end):
                    res[i] = f'I-{classifier_class}'
        return torch.Tensor([self.tags(token) for token in res]).to(x.device)
    
    def smart_forward(self, x: torch.Tensor, x_chars: torch.Tensor, s_ids: torch.Tensor = None) -> torch.Tensor:
        batch = x.shape[0]
        res = torch.Tensor(x.shape)
        for bi in range(batch):
            curr_batch = x[bi]
            curr_batch_tags = self.smart_forward_single(curr_batch)
            res[bi] = curr_batch_tags
        
        return res

    def add_example(self, sentence: torch.Tensor, tags: torch.Tensor) -> None:
        if self.smart:
            self.smart_add_example(sentence, tags)
        else:
            self.classifier[sentence, tags] += 1
    
    def smart_add_single(self, sentence: torch.Tensor, tags: torch.Tensor) -> None:
        sentence_decode = self.vocab.decode(sentence)
        tags_decode = self.tags.decode(tags)
        ents = get_ents(sentence_decode, tags_decode)
        self.dictionary.extend(ents)
    
    def smart_add_example(self, sentence: torch.Tensor, tags: torch.Tensor) -> None:
        batch = sentence.shape[0]
        for bi in range(batch):
            curr_batch = sentence[bi]
            curr_tags = tags[bi]
            self.smart_add_single(curr_batch, curr_tags)
    
    def forward(self, x: torch.Tensor, x_chars: torch.Tensor, s_ids: torch.Tensor = None) -> torch.Tensor:
        '''
        Given a sentence x, returns a softmax distribution
        over the space

        x has shape (batch, sequence_length)

        returns (batch, sequence length, tag_vocab_size) of all the tags
        '''
        if self.smart:
            return self.smart_forward(x, x_chars, s_ids)
        # x (batch, s)
        # counts (vocab, tag)
        # expected (batch, s, tag)
        # s is the index in vocab to select
        counts = self.classifier[x.long()]
        # (batch size, sequence length, tag_vocab)
        # return F.softmax(counts, dim=-1)
        return torch.argmax(counts, dim=2)
    
    def compute_uncertainty(self, x: torch.Tensor, x_chars: torch.Tensor, s_ids: torch.Tensor = None) -> torch.Tensor:
        batch_size = x.shape[0]
        return torch.zeros(batch_size)


class PhraseDictionaryModel(DictionaryModel):
    def __init__(
        self,
        vocab,
        tags,
    ):
        super(PhraseDictionaryModel, self).__init__(vocab, tags, smart=True)
