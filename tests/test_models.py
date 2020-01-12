import unittest
from typing import (
    List
)
from tqdm import tqdm
import torch
from torch import nn

import ner
from ner.models import (
    bilstm_crf,
    dictionary_model,
    elmo_bilstm_crf,
    cached_bilstm_crf,
)
from ner import utils as ner_utils
from ner import conlldataloader

from . import utils

class TestModels(unittest.TestCase):
    
    def _test_single_model_train(self, model: nn.Module):
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

        if not isinstance(model, dictionary_model.DictionaryModel):
            optim = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-9)
        else:
            optim = None

        # iterate over epochs
        for e in range(1):
            loss_sum = 0

            with tqdm(data_loader) as pbar:
                for i, (s_ids, x, x_chars, y, weight) in enumerate(pbar):
                    if isinstance(model, dictionary_model.DictionaryModel):
                        model.add_example(x.long(), y.long())
                        continue
                    model.zero_grad()
                    model_loss = model.compute_mle(x, x_chars, y)
                    loss = torch.mean(model_loss)
                    loss.backward() # backpropogate
                    optim.step() # update parameters
    
    def _test_single_model_eval(self, model: nn.Module):
        vocab = utils.build_sample_vocab()
        tag_vocab = utils.build_sample_tag_vocab()
        train_dataset = utils.construct_sample_dataloader()
        data_loader = conlldataloader.get_data_loader(
            vocab,
            tag_vocab,
            train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        f1_data, _ = ner_utils.compute_f1_dataloader(
            model=model,
            data_loader=data_loader,
            tag_vocab=tag_vocab,
        )

        ner_utils.estimate_f1(
            model=model,
            data_loader=data_loader,
            tag_vocab=tag_vocab,
            threshold=1,
        )

        # compute average
        avg = ner_utils.compute_avg_f1(f1_data)
    
    def test_shapes(self, batch_size=1):
        models = utils.build_all_models()
        vocab = utils.build_sample_vocab()
        tag_vocab = utils.build_sample_tag_vocab()
        train_dataset = utils.construct_sample_dataloader()

        data_loader = conlldataloader.get_data_loader(
            vocab,
            tag_vocab,
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        def _convert(item: object) -> torch.Tensor:
            if isinstance(item, torch.Tensor):
                return item
            else:
                return torch.Tensor(item)

        for batch_i, (s_ids, x, x_chars, y, weight) in enumerate(data_loader):
            outs = [_convert(model(x, x_chars)) for model in models]
            for i in range(1, len(outs)):
                # all models should output the same dimensions
                assert outs[i].shape == outs[i - 1].shape
    
    def test_train_models(self):
        models = utils.build_all_models()
        for model in models:
            self._test_single_model_train(model)
    
    def test_eval_models(self):
        models = utils.build_all_models()
        for model in models:
            self._test_single_model_eval(model)
    
    def test_analyze(self):
        models = utils.build_all_models()
        vocab = utils.build_sample_vocab()
        tag_vocab = utils.build_sample_tag_vocab()
        train_dataset = utils.construct_sample_dataloader()

        data_loader = conlldataloader.get_data_loader(
            vocab,
            tag_vocab,
            train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        for model in models:
            ner_utils.analyze_predictions(
                model=model,
                data_loader=data_loader,
                vocab=vocab,
                tag_vocab=tag_vocab,
                log_file=utils.MockFile(),
                csv_file=None, # no need to output to csv
            )
    
    def test_elmo_from_raw_sentence(self):
        vocab = utils.build_sample_vocab()
        tag_vocab = utils.build_sample_tag_vocab()
        train_dataset = utils.construct_sample_dataloader()
        embedding_dim = 4
        hidden_dim = 4
        batch_size = 4

        elmo_model = elmo_bilstm_crf.ELMo_BiLSTM_CRF(vocab, tag_vocab, hidden_dim, batch_size)

        data_loader = conlldataloader.get_data_loader(
            vocab,
            tag_vocab,
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        for batch_i, (s_ids, x, x_chars, y, weight) in enumerate(data_loader):
            # assure no errors in elmo reconstruction
            elmo_model(x, None)
            break
        
    def test_cached_model(self):
        dataset = utils.construct_sample_unlabeled_dataset()
        vocab = utils.build_sample_vocab()
        tag_set = utils.build_sample_tag_vocab()
        ce = utils.construct_cached_embedder()
        cbc = ner.utils.build_model(
            model_type='cached',
            embedding_dim=1024,
            hidden_dim=300,
            vocab=vocab,
            tag_vocab=tag_set,
            batch_size=2,
        )

        cbc.embedder = ce

        data_loader = conlldataloader.get_unlabeled_data_loader(
            vocab=vocab,
            categories=tag_set,
            unlabeled_data=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=conlldataloader.collate_unlabeld_fn_with_sid,
        )

        for s_ids, sentence, sentence_chars in data_loader:
            embedded_sent = ce.batched_forward_cached(s_ids, sentence)
            computed_embedded_sent = ce.forward(sentence_chars)
            computed_embedded_sent_2 = ce.forward(sentence_chars)
            assert len(sentence[0]) == len(embedded_sent[0])
            assert embedded_sent.shape == computed_embedded_sent.shape

        data_loader = conlldataloader.get_unlabeled_data_loader(
            vocab=vocab,
            categories=tag_set,
            unlabeled_data=dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=conlldataloader.collate_unlabeld_fn_with_sid,
        )

        cbc.eval()
        for s_ids, sentence, sentence_chars in data_loader:
            torch.random.manual_seed(0)
            res = cbc(sentence, sentence_chars, s_ids)
            torch.random.manual_seed(0)
            res2 = cbc(sentence, sentence_chars, None)
        
        cbc.train()
        self._test_single_model_train(model=cbc)
        
        cbc.eval()
        labels = ner_utils.compute_labels(
            model=cbc,
            data_loader=data_loader,
            tag_vocab=tag_set,
            verbose=False,
        )

        print(labels)