import unittest
import torch
from torch import nn


from ner.trainer import Trainer

from . import utils

class TestTrainer(unittest.TestCase):

    def test_trainer(self):
        models = utils.build_all_models()
        for model in models:
            self._test_train_model(model)
    
    def test_invalid_opt(self):
        model = utils.build_all_models()[-1]
        try:
            self._test_train_model(model, 'UnknownOptimizer')
            assert False # pragma: no cover
        except:
            pass

    def test_adam_trainer(self):
        model = utils.build_all_models()[-1]
        self._test_train_model(model, 'ADAM')

    def _test_train_model(self, model: nn.Module, optimizer_type='SGD'):
        logger = utils.MockLogger()
        trainer = Trainer(
            model=model,
            learning_rate=0.01,
            weight_decay=1e-9,
            momentum=0.0,
            optimizer_type=optimizer_type,
            vocab=utils.build_sample_vocab(),
            tags=utils.build_sample_tag_vocab(),
            batch_size=2,
            shuffle=False,
            num_workers=0,
            train_dataset=utils.construct_sample_dataloader(),
            test_dataset=utils.construct_sample_dataloader(),
            logger=logger,
            device='cpu',
            verbose_log=True,
            verbose_print=True,
        )

        best_epoch, best_epoch_summary = trainer.train(2)

        # check best epoch range
        assert best_epoch >= 0 and best_epoch < 2

        # check best_epoch_summary for the following structure
        epoch_summary_def = {
            'train_loss_avg' : float,
            'valid_accuracy' : float,
            'train_accuracy' : float,
            'valid_f1_avg' : float,
            'train_f1_avg' : float,
            'valid_f1' : dict,
            'train_f1' : dict,
            'epoch_number': int,
        }

        assert type(best_epoch_summary) == dict
        assert len(best_epoch_summary) == len(epoch_summary_def)
        for key, t in epoch_summary_def.items():
            assert key in best_epoch_summary and type(best_epoch_summary[key]) == t
        
        # make sure a model is returned
        assert isinstance(trainer.get_best_model(), nn.Module)

        # make sure the best model is the same model as the one passed in
        assert type(trainer.get_best_model()) == type(model)

        # assert that the training progress shows 2 epochs
        assert len(trainer.get_progress()) == 2

        logger.flush()
