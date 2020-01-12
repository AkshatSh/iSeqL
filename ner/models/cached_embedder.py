import torch
from torch import nn
from tqdm import tqdm

from typing import (
    List,
    Dict,
    Tuple,
)

from ner import constants
from ner import conlldataloader
from overrides import overrides

class CachedEmbedder(nn.Module):
    '''
    This is an embedder class that 
    '''
    def __init__(
        self,
        embedder: nn.Module,
        embedding_dimensions: int,
    ):
        super(CachedEmbedder, self).__init__()
        self.embedder = embedder
        self.embedding_dimensions = embedding_dimensions
    
        self.index_map = []
        self.database = None
        self.sentence_start = {}
        self.sentence_end = {}
    
    def cache_dataset(
        self,
        dataset: conlldataloader.ConllDataSetUnlabeled,
        device: str = 'cpu',
        verbose: bool = False
    ):
        '''
        Given a dataset and device cache the dataset so that the embedder
        does not need to be rerun

        args:
            dataset: unlabeleddataset
            device: the device to run the caching process on
            verbose: show progress bars
        '''
        generator = dataset
        if verbose:
            generator = tqdm(dataset)
        
        total = 0
        for s_id, sent in generator:
            self.sentence_start[s_id] = total
            embeded_sentence = self.embedder.get_embedding_from_sentence(sent, device)
            assert len(sent) == len(embeded_sentence)

            self.database = torch.cat((self.database, embeded_sentence)) if self.database is not None else embeded_sentence
            self.index_map.extend(
                [(word, sent, s_id) for word in sent]
            )

            total += len(sent)
            self.sentence_end[s_id] = total
    
    def forward_cached(self, s_id: int, sent: List, verify: bool = False) -> torch.Tensor:
        '''
        Given a sentence id and a sentence to verify with
        retrieve the embedding from the cache

        args:
            s_id: int, id of sentence
            sent: an iteratable to verify length of embedding with
        return:
            torch.Tensor: embedding of sentence
        '''
        start_idx: int = self.sentence_start[s_id]
        end_idx: int = self.sentence_end[s_id]
        embeded_sentence: torch.Tensor = self.database[start_idx : end_idx]
        # info = self.index_map[start_idx : end_idx]
        assert not verify or len(sent) == len(embeded_sentence)
        res = embeded_sentence
        return res.detach()

    def batched_forward_cached(
        self,
        s_id: torch.Tensor,
        sent: torch.Tensor,
    ) -> torch.Tensor:
        '''
        get a batch of embedded sentences
        '''
        output = torch.zeros((sent.shape[0], sent.shape[1], self.embedder.DIMENSIONS)).to(sent.device)
        for i, (c_id, c_sent) in enumerate(zip(s_id, sent)):
            embed_sent = self.forward_cached(c_id.item(), c_sent, verify=False)
            output[i, :len(embed_sent)] = embed_sent
        return output

    
    def forward(self, sentence_chars: torch.Tensor) -> torch.Tensor:
        '''
        run the sentence chars argument through the embedder and return
        result
        '''
        tensor = self.embedder.forward(sentence_chars)['elmo_representations'][0]
        return tensor.detach()
    
    def dimensions(self):
        '''
        get the number of embedding dimensions
        '''
        return self.embedding_dimensions
    
    def save(self, save_dict: Dict[str, object], key: str):
        internal_state = (
            self.index_map,
            self.database,
            self.sentence_start,
            self.sentence_end,
            self.embedding_dimensions,
        )

        save_dict[key] = internal_state
    
    def load(self, load_dict: Dict[str, object], key: str):
        internal_state = load_dict[key]
        (
            self.index_map,
            self.database,
            self.sentence_start,
            self.sentence_end,
            self.embedding_dimensions,
        ) = internal_state
    
    @overrides
    def _apply(self, fn):
        if self.database is not None:
            self.database = fn(self.database)
        return super()._apply(fn)