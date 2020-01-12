import unittest
import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

from ner.models.elmo import FrozenELMo
from ner.trainer import Trainer

from . import utils

FIXED_SEED = 1234

class TestFrozenELMo(unittest.TestCase):
    def test_singleton(self):
        '''
        2 calls to get instance should be the exact same object
        '''

        inst1 = FrozenELMo.instance()
        inst2 = FrozenELMo.instance()

        assert id(inst1) == id(inst2)
    
    def test_forward(self):
        '''
        Ensures that the weights do not update in a simple network
        '''

        # reproducibility
        torch.manual_seed(FIXED_SEED)

        class Simple(nn.Module):
            def __init__(self):
                super(Simple, self).__init__()
                self.embedder = FrozenELMo.instance()
                self.linear = nn.Linear(FrozenELMo.DIMENSIONS, 1)
            
            def forward(self, x, x_chars, s_ids=None):
                return F.relu(self.linear(
                    F.tanh(self.embedder(
                        x_chars
                    )['elmo_representations'][0])
                ))
            
            def compute_mle(self, x, x_chars, y, s_ids=None):
                feats = self(x, x_chars)
                rand_out = torch.randn(feats.shape)
                return F.mse_loss(feats, rand_out)
        
        model = Simple()
        initial_embedder = copy.deepcopy(model.embedder)
        initial_linear_layer = copy.deepcopy(model.linear)
        trainer = Trainer(
            model=model,
            learning_rate=0.01,
            weight_decay=1e-9,
            momentum=0.0,
            optimizer_type='SGD',
            vocab=utils.build_sample_vocab(),
            tags=utils.build_sample_tag_vocab(),
            batch_size=2,
            shuffle=False,
            num_workers=0,
            train_dataset=utils.construct_sample_dataloader(),
            test_dataset=utils.construct_sample_dataloader(),
            logger=utils.MockLogger(),
            device='cpu',
            verbose_log=False,
            verbose_print=False,
        )

        trainer.train(2)

        assert utils.compare_models(model.embedder, initial_embedder)
        assert not utils.compare_models(model.linear, initial_linear_layer)


            