import unittest
import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from ner.models.elmo import FrozenELMo
from ner.trainer import Trainer
from ner import conlldataloader

from . import utils

FIXED_SEED = 1234

class TestDictionary(unittest.TestCase):
    def test_dictionary(self):
        '''
        2 calls to get instance should be the exact same object
        '''

        model = utils.build_all_models(-1)[0]
        vocab = utils.build_sample_vocab()
        tag_vocab = utils.build_sample_tag_vocab()
        train_dataset = utils.construct_sample_dataloader()


        data_loader = conlldataloader.get_data_loader(
            vocab,
            tag_vocab,
            train_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )

        # iterate over epochs
        for e in range(1):
            with tqdm(data_loader) as pbar:
                for i, (s_ids, x, x_chars, y, weight) in enumerate(pbar):
                    model.add_example(x.long(), y.long())
        
        test_examples = [
            ['one', 'word', 'here'],
            ['one', 'two', 'word', 'here'],
            ['one', 'word', 'here'],
            ['a', 'completely', 'unrelated', 'sentence'],
        ]

        for example in test_examples:
            encoded = torch.Tensor([vocab(e) for e in example])
            res = model.smart_forward_single(encoded)
        


            