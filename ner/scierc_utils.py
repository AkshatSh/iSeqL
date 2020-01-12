import pickle
import os
from typing import Tuple

try: # pragma: no cover
    import constants
    import vocab
    from vocab import Vocab
    from scierc_dataloader import SCIERCDataset
except: # pragma: no cover
    from . import (
        constants,
        vocab,
        scierc_dataloader,
    )
    from .vocab import Vocab
    from .scierc_dataloader import SCIERCDataset


SCIERCLoadedDatasetType = Tuple[
    SCIERCDataset,
    SCIERCDataset,
    Vocab,
    Vocab,
]

SCIERCConstantsListType = Tuple[
    str, # train conll dataset
    str, # valid conll dataset
    str, # train conll data_loader
    str, # valid conll data_loader
    str, # vocab
    str, # tag vocab
]

def get_preprocessed_constants() -> SCIERCConstantsListType:
    return (
        constants.SCIERC_CONLL_PROCESSED_TRAIN_DATASET,
        constants.SCIERC_CONLL_PROCESSED_VALID_DATASET,
        constants.SCIERC_PROCESSED_TRAIN_DATALOADER,
        constants.SCIERC_PROCESSED_VALID_DATALOADER,
        constants.SCIERC_PROCESSED_VOCAB,
        constants.SCIERC_PROCESSED_TAGS,
    )

def get_normal_constants() -> SCIERCConstantsListType:
    return (
        constants.SCIERC_CONLL_DATASET,
        constants.SCIERC_CONLL_VALID_DATASET,
        constants.SCIERC_DATALOADER,
        constants.SCIERC_VALID_DATALOADER,
        constants.SCIERC_VOCAB,
        constants.SCIERC_TAGS,
    )

def get_constants() -> SCIERCConstantsListType:
    if constants.SCIERC_USE_PREPROCESSED:
        return get_preprocessed_constants()
    else:
        return get_normal_constants()

def load() -> SCIERCLoadedDatasetType:
    '''
    Conll2003 Data loader
    '''
    _, _, train_dataloader, valid_dataloader, vocab_file, tags_file = get_constants()
    print('loading SCIERC all datasets')
    with open(train_dataloader, 'rb') as f:
        train_dataset = pickle.load(f)
    
    with open(valid_dataloader, 'rb') as f:
        valid_dataset = pickle.load(f)

    with open(vocab_file, 'rb') as f:
        train_vocab = pickle.load(f)

    with open(tags_file, 'rb') as f:
        output_categories = pickle.load(f)

    return train_dataset, valid_dataset, train_vocab, output_categories

def save(
    train_dataset: SCIERCDataset,
    valid_dataset: SCIERCDataset,
    train_vocab: Vocab,
    output_categories: Vocab,
) -> None:
    _, _, train_dataloader, valid_dataloader, vocab_file, tags_file = get_constants()
    '''
    Saves the conll dataset to files
    '''
    with open(train_dataloader, 'wb') as f:
        pickle.dump(train_dataset, f)
    
    with open(valid_dataloader, 'wb') as f:
        pickle.dump(valid_dataset, f)

    with open(vocab_file, 'wb') as f:
        pickle.dump(train_vocab, f)

    with open(tags_file, 'wb') as f:
        pickle.dump(output_categories, f)


def create_datasets() -> SCIERCLoadedDatasetType:
    train_conll, valid_conll, _, _, _, _ = get_constants()
    train_dataset = SCIERCDataset(train_conll)
    valid_dataset = SCIERCDataset(valid_conll)

    print('processing train dataset')
    train_dataset.parse_file()
    print('finished processing train dataset')

    print('processing valid dataset')
    valid_dataset.parse_file()
    print('finished processing valid dataset')

    print('build training vocab')
    train_vocab = vocab.build_vocab(train_dataset.word_list)
    print('done building vocab')

    print('build output vocab')
    output_categories = vocab.build_output_vocab(train_dataset.categories)
    print('done building output vocab')

    return train_dataset, valid_dataset, train_vocab, output_categories

def clean_saved() -> None:
    '''
    Remove all the saved files
    '''
    _, _, train_dataloader, valid_dataloader, vocab, tags = get_constants()
    os.remove(train_dataloader)
    os.remove(valid_dataloader)
    os.remove(vocab)
    os.remove(tags)