from typing import (
    List,
    Tuple,
    Dict,
)
import torch
from torch import nn

from ner.conlldataloader import (
    ConllDataSet,
    ConllDataSetUnlabeled,
    ConllTorchDataset,
)

from ner.vocab import (
    Vocab,
    build_vocab,
    build_output_vocab,
    BinaryVocab,
)

from ner import constants
import ner.utils

from ner.models import (
    bilstm_crf,
    dictionary_model,
    elmo_bilstm_crf,
    cached_embedder,
    cached_bilstm_crf,
    elmo,
)

SAMPLE_DATASET = [
    (
        ['one', 'word', 'here'],
        ['B-tag', 'I-tag', 'O'],
    ),
    (
        ['one', 'two', 'word', 'here'],
        ['O', 'B-tag', 'O', 'O'],
    ),
    (
        ['one', 'word', 'here'],
        ['B-tag', 'I-tag', 'O'],
    ),
    (
        ['a', 'completely', 'unrelated', 'sentence'],
        ['O', 'O', 'O', 'O'],
    )
]

TEST_UTILS_CACHE = {}

def get_cached_instance(key: str, constructor: callable) -> object:
    if key not in TEST_UTILS_CACHE:
        TEST_UTILS_CACHE[key] = constructor()
    return TEST_UTILS_CACHE[key]

def construct_dataloader(dataset: List[Tuple[List[str], List[str]]]) -> ConllDataSet:
    # mock a conlldataloader object
    conlldataset = ConllDataSet(file_name=None)
    for (input_point, output_point) in dataset:
        conlldataset.word_list.extend(input_point)
        input_entry = [{constants.CONLL2003_WORD: word} for word in input_point]
        conlldataset.data.append({
            'input': input_entry,
            'output': output_point,
        })
        conlldataset.categories.extend(output_point)
    return conlldataset

def construct_cached_embedder() -> cached_embedder.CachedEmbedder:
    def _construct_cached_embedder() -> cached_embedder.CachedEmbedder:
        dataset = construct_sample_unlabeled_dataset()
        ce = cached_embedder.CachedEmbedder(elmo.FrozenELMo.instance(), elmo.FrozenELMo.DIMENSIONS)
        ce.cache_dataset(dataset=dataset)
        return ce
    return get_cached_instance('unlabeled_cached_embedder', _construct_cached_embedder)

def construct_sample_dataloader() -> ConllDataSet:
    def _construct_sample_dataloader() -> ConllDataSet:
        return construct_dataloader(SAMPLE_DATASET)
    return get_cached_instance('sample_dataloader', _construct_sample_dataloader)

def construct_sample_unlabeled_dataset() -> ConllDataSetUnlabeled:
    def _construct_sample_unlabeled_dataset() -> ConllDataSetUnlabeled:
        return ConllDataSetUnlabeled(construct_sample_dataloader())
    return get_cached_instance('sample_unlabeled_dataloader', _construct_sample_unlabeled_dataset)

def build_sample_vocab() -> Vocab:
    def _construct_vocab() -> Vocab:
        dl = construct_dataloader(SAMPLE_DATASET)
        return build_vocab(dl.word_list)
    return get_cached_instance('sample_vocab', _construct_vocab)

def build_sample_tag_vocab() -> Vocab:
    def _construct_tag_vocab() -> Vocab:
        dl = construct_dataloader(SAMPLE_DATASET)
        return build_vocab(dl.categories)
    return get_cached_instance('sample_tag_vocab', _construct_tag_vocab)

def build_sample_binary_vocab() -> BinaryVocab:
    vocab = build_sample_tag_vocab()
    return BinaryVocab(vocab, select_class='tag', default_value='O')

def build_sample_torch_dataset() -> ConllTorchDataset:
    def _construct_torch_dataset() -> ConllTorchDataset:
        dataset = construct_sample_dataloader()
        vocab = build_sample_vocab()
        tag_vocab = build_sample_tag_vocab()

        return ConllTorchDataset(
            vocab=vocab,
            categories=tag_vocab,
            conlldataset=dataset,
        )
    return get_cached_instance('sample_torch_dataset', _construct_torch_dataset)

def compare_models(model1: nn.Module, model2: nn.Module) -> bool:
    '''
    Check to see if two models have the exact same parameters value
    https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/2
    '''
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

def build_all_models(index: int=None) -> List[nn.Module]:
    '''
    Build all models to test
    '''
    vocab = build_sample_vocab()
    tag_vocab = build_sample_tag_vocab()
    embedding_dim = 4
    hidden_dim = 4
    batch_size = 4

    model_types = ['dictionary', 'elmo_bilstm_crf', 'bilstm_crf', 'phrase_dictionary']
    if index is not None:
        model_types = [model_types[index]]

    models = [ner.utils.build_model(
        model_type=model_type,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        vocab=vocab,
        tag_vocab=tag_vocab,
        batch_size=batch_size,
    ) for model_type in model_types]

    models.append(elmo_bilstm_crf.ELMo_BiLSTM_CRF(vocab, tag_vocab, hidden_dim, batch_size, freeze_elmo=False))


    # models = [
    #     bilstm_crf.BiLSTM_CRF(vocab, tag_vocab, embedding_dim, hidden_dim, batch_size),
    #     elmo_bilstm_crf.ELMo_BiLSTM_CRF(vocab, tag_vocab, hidden_dim, batch_size, freeze_elmo=True),
    #     elmo_bilstm_crf.ELMo_BiLSTM_CRF(vocab, tag_vocab, hidden_dim, batch_size, freeze_elmo=False),
    #     dictionary_model.DictionaryModel(vocab, tag_vocab),
    # ]

    return models

class MockModel(nn.Module):
    '''
    Import a mocked model for a torch model,
    always returns a tensor of [1x1] containing random_val
    '''
    def __init__(self, random_val: int = 42):
        super(MockModel, self).__init__()
        self.random_val = random_val
        self.linear = nn.Linear(random_val, 1)
    
    def forward(self, *args, **kwargs):
        return torch.Tensor([self.random_val])
    
    # for CRF models
    def compute_uncertainty(self, *args, **kwargs):
        return self(args, kwargs)

class MockFile(object):
    def __init__(
        self,
        write_func: callable = lambda x: x,
    ):
        super(MockFile, self).__init__()
        self.write_func = write_func
    
    def write(self, x):
        return self.write_func(x)

class MockLogger(object):
    def __init__(
        self,
    ):
        super(MockLogger, self).__init__()
    
    def scalar_summary(self, *args, **kwargs):
        pass
    
    def flush(self, *args, **kwargs):
        pass