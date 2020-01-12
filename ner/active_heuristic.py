import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import tqdm
from allennlp.modules.elmo import batch_to_ids
import faiss
import pickle
from typing import (
    List,
    Dict,
    Tuple,
)
import random
import numpy as np
from nltk.corpus import stopwords 

try: # pragma: no cover
    from models import crf
    from models.elmo import FrozenELMo
    import conlldataloader
    import vocab
    import utils
except: # pragma: no cover
    from . import (
        conlldataloader,
        vocab,
        utils,
    )
    from .models.elmo import FrozenELMo
    from .models import crf

STOP_WORDS = set(stopwords.words('english'))

class ActiveHeuristic(object):
    '''
    An abstract class for different active learning heuristics
    '''
    def __init__(self):
        pass
    
    '''
    Any preprocessing the active learning heuristic needs to do
    to prepare the heuristic
    '''
    def prepare(
        self,
        model: nn.Module,
        dataset: conlldataloader.ConllDataSetUnlabeled,
        device: str = 'cpu'
    ) -> None:
        pass
    
    def save(
        self,
        dir_name: str,
    ): # pragma: no cover
        pass
    
    def load(
        self,
        dir_name: str,
    ) -> bool: # pragma: no cover
        return True
    
    '''
    Returns a tensor with each index corresponding to the probability,
    that that index should be sampled
    '''
    def evaluate(
        self,
        model: torch.nn.Module,
        dataset: conlldataloader.ConllDataSetUnlabeled,
        device: str = 'cpu',
    ) -> torch.Tensor:
        raise NotImplementedError()

class Random(ActiveHeuristic):
    def __init__(self, vocab: vocab.Vocab, tag_vocab: vocab.Vocab):
        self.vocab = vocab
        self.tag_vocab = tag_vocab
    
    def evaluate(
        self,
        model: torch.nn.Module,
        dataset: conlldataloader.ConllDataSetUnlabeled,
        device: str = 'cpu',
    ) -> torch.Tensor:
        return F.softmax(torch.zeros(len(dataset)), dim=0)

class Uncertantiy(ActiveHeuristic):
    def __init__(
        self,
        vocab: vocab.Vocab,
        tag_vocab: vocab.Vocab,
    ):
        super(Uncertantiy, self).__init__()
        self.vocab = vocab
        self.tag_vocab = tag_vocab
    
    def evaluate(
        self,
        model: crf.CRF,
        dataset: conlldataloader.ConllDataSetUnlabeled,
        device: str ='cpu',
    ) -> torch.Tensor:
        model.eval()
        dataloader = self._build_data_loader(dataset)
        output = torch.zeros(len(dataset)).to(device)
        for i, (s_ids, x, x_chars) in enumerate(tqdm.tqdm(dataloader, total=len(dataloader))):
            x, x_chars, s_ids = x.to(device), x_chars.to(device), torch.Tensor([s_ids]).to(device)
            x = x.unsqueeze(0)
            x_chars = x_chars.unsqueeze(0)
            s_ids = s_ids.unsqueeze(0)
            score = model.compute_uncertainty(x, x_chars, s_ids)
            output[i] = score
        output = F.softmax(output, dim=0)
        model.train()
        return output
    
    def _build_data_loader(
        self,
        dataset: conlldataloader.ConllDataSetUnlabeled,
    ) -> data.dataset:
        return conlldataloader.ConllTorchDatasetUnlabeld(
            self.vocab,
            self.tag_vocab,
            dataset,
        )

class KNNEmbeddings(ActiveHeuristic):
    def __init__(
        self,
        vocab: vocab.Vocab,
        tag_vocab: vocab.Vocab,
    ):
        super(KNNEmbeddings, self).__init__()
        self.vocab = vocab
        self.tag_vocab = tag_vocab
        self.embedder = FrozenELMo.instance()
    
    def _prepare_faiss(
        self,
        dataset: conlldataloader.ConllDataSetUnlabeled,
        device: str = 'cpu'
    ):
        self.embedder.to(device)
        self.embedder.eval()
        dimensions = 1024

        database = None
        index_map = []
        sentence_start = {}
        sentence_end = {}
        total = 0

        for ind, sentence in tqdm.tqdm(dataset):
            sentence_start[ind] = total
            # sentence: [seq_len] of strings
            embeded_sentence = self.embedder.get_embedding_from_sentence(sentence, device)

            # embeded_sentence = [seq_len x dimensions] of floats
            assert len(sentence) == len(embeded_sentence)
            for i, embedding in enumerate(embeded_sentence):
                if sentence[i] in STOP_WORDS:
                    # super far away, should not be selected
                    embeded_sentence[i, :] = float('inf')
            database = torch.cat((database, embeded_sentence)) if database is not None else embeded_sentence

            index_map.extend(
                [(word, sentence, ind) for word in sentence]
            )

            total += len(sentence)
            sentence_end[ind] = total

        self.index_map = index_map
        self.database = database
        self.sentence_start = sentence_start
        self.sentence_end = sentence_end
    
    def prepare(
        self,
        model: nn.Module,
        dataset: conlldataloader.ConllDataSetUnlabeled,
        device: str = 'cpu'
    ) -> None:
        self._prepare_faiss(dataset, device)
        self._prepare_faiss_index()
    
    def evaluate_with_labeled(
        self,
        model: crf.CRF,
        dataset: conlldataloader.ConllDataSetUnlabeled,
        labeled_indexes: List[int],
        labeled_points: List[Tuple[List[str], List[str]]],
        device: str ='cpu',
    ) -> None:
        '''
        New Active Learning Algorithm:
            [x] Convert the labeled sentence to their contextualized embeddings
            [x] Sample some points across all different classes
            [x] Run some similarity search metric (FAISS)
            [x] Grab K nearest neighbors from each query point (candidates)
            [x] Score each unlabeled sentence, by the number of candidates in that sentence
            [x] Softmax it for probability distribution

        '''
        self.database.to(device)
        labeled_embeddings = [
            self.database[
                self.sentence_start[s_id]:self.sentence_end[s_id]
            ] for s_id in labeled_indexes
        ]

        # map tag -> [(word, embedding)]
        class_map = {}
        for s_id, (_, s_in, s_out), embeddings in zip(labeled_indexes, labeled_points, labeled_embeddings):
            for (word, tag, embedding) in zip(s_in, s_out, embeddings):
                if word in STOP_WORDS or tag == 'O': # pragma: no cover
                    # ignore STOP_WORDS and ignore negative labels
                    continue
                removed_bio_tag = utils.remove_bio_input(tag)
                if removed_bio_tag not in class_map:
                    class_map[removed_bio_tag] = []
                class_map[removed_bio_tag].append((word, embedding))
        

        database_count = np.zeros((len(self.database)))
        candidates = []
        for tag, potential_query in class_map.items():
            sample_size = min(len(potential_query), 100)

            # only sample at most 10 items for the query
            query = random.sample(potential_query, sample_size)
            embedded_query = torch.stack([embedding for (_, embedding) in query])

            # distances from all query embedded vectors
            distances, indexes = self.index.search(embedded_query.numpy(), 5)

            # create a list of values associated with each result
            index_values = (np.arange(5) + 1)[::-1] * (np.zeros(indexes.shape) + 1)
            
            for ind, val in zip(indexes.flatten(), index_values.flatten()):
                database_count[ind] = val
        
        # candidates contains an int specifying all the potential candidates
        # invovled in the sample
        output = torch.zeros(len(dataset)).to(device)
        for i, (s_id, s_in) in enumerate(dataset):
            output[i] = database_count[self.sentence_start[s_id]:self.sentence_end[s_id]].mean()
        output = F.softmax(output, dim=0)
        return output
    
    def evaluate(
        self,
        model: crf.CRF,
        dataset: conlldataloader.ConllDataSetUnlabeled,
        device: str ='cpu',
    ) -> torch.Tensor:
        return None
    
    def save(
        self,
        dir_name: str,
    ): # pragma: no cover
        internal_data = (
            self.index_map,
            self.database.cpu().numpy() if self.database is not None else self.database,
            self.sentence_start,
            self.sentence_end,
        )

        with open(os.path.join(dir_name, "KNNEmbeddingActiveLearning.pkl"), 'wb') as f:
            pickle.dump(internal_data, f)
    
    def load(
        self,
        dir_name: str,
    ) -> bool: # pragma: no cover
        internal_data_path = os.path.join(dir_name, "KNNEmbeddingActiveLearning.pkl")
        if not os.path.exists(internal_data_path):
            return False
        with open(internal_data_path, 'rb') as f:
            internal_data = pickle.load(f)
            self.index_map, self.database, self.sentence_start,self.sentence_end = internal_data
            self._prepare_faiss_index()
        return True
    
    def _database_numpy(self):
        if isinstance(self.database, torch.Tensor):
            self.database = self.database.cpu().numpy()
    
    def _database_tensor(self):
        if isinstance(self.database, np.ndarray):
            self.database = torch.Tensor(self.database)
    
    def _normalize_database(self):
        faiss.normalize_L2(self.database)

    def _prepare_faiss_index(self):
        self.index = faiss.IndexFlatIP(FrozenELMo.DIMENSIONS)
        if self.database is None: # pragma: no cover
            return
        self._database_numpy()
        self._normalize_database()
        self.index.add(self.database)
        self._database_tensor()
