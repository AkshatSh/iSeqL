import pickle
import os

try: # pragma: no cover
    import constants
    import conlldataloader
    import vocab
except: # pragma: no cover
    from . import (
        constants,
        conlldataloader,
        vocab,
    )

def load():
    '''
    Conll2003 Data loader
    '''
    print('loading CONLL all datasets')
    with open(constants.SAVED_CONLL_DATALOADER, 'rb') as f:
        train_dataset = pickle.load(f)

    with open(constants.SAVED_CONLL_VALID_DATALOADER, 'rb') as f:
        valid_dataset = pickle.load(f)

    with open(constants.SAVED_CONLL_VOCAB, 'rb') as f:
        train_vocab = pickle.load(f)

    with open(constants.SAVED_CONLL_CATEGORIES, 'rb') as f:
        output_categories = pickle.load(f)

    return train_dataset, valid_dataset, train_vocab, output_categories

def save(train_dataset, valid_dataset, train_vocab, output_categories):
    '''
    Saves the conll dataset to files
    '''
    with open(constants.SAVED_CONLL_DATALOADER, 'wb') as f:
        pickle.dump(train_dataset, f)

    with open(constants.SAVED_CONLL_VALID_DATALOADER, 'wb') as f:
        pickle.dump(valid_dataset, f)

    with open(constants.SAVED_CONLL_VOCAB, 'wb') as f:
        pickle.dump(train_vocab, f)

    with open(constants.SAVED_CONLL_CATEGORIES, 'wb') as f:
        pickle.dump(output_categories, f)


def create_datasets():
    train_dataset = conlldataloader.ConllDataSet(constants.CONLL2003_TRAIN)
    valid_dataset = conlldataloader.ConllDataSet(constants.CONLL2003_VALID)

    print('processing train dataset')
    train_dataset.parse_file()
    print('finished processing train dataset')

    print('build training vocab')
    train_vocab = vocab.build_vocab(train_dataset.word_list)
    print('done building vocab')

    print('build output vocab')
    output_categories = vocab.build_output_vocab(train_dataset.categories)
    print('done building output vocab')

    print('processing valid dataset')
    valid_dataset.parse_file()
    print('finished processing valid dataset')

    return train_dataset, valid_dataset, train_vocab, output_categories

def clean_saved():
    '''
    Remove all the saved files
    '''
    os.remove(constants.SAVED_CONLL_DATALOADER)
    os.remove(constants.SAVED_CONLL_VOCAB)
    os.remove(constants.SAVED_CONLL_VALID_DATALOADER)
    os.remove(constants.SAVED_CONLL_CATEGORIES)
    os.rmdir(constants.SAVED_DIR)