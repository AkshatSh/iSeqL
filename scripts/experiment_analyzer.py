# typing
from typing import (
    List,
    Dict,
    Tuple,
)

# system imports
import os
import sys
import pickle
import argparse
import csv

# library imports
import torch 
from torch import nn
import numpy
import matplotlib.pyplot as plt

# local library imports
import ner
import ner.vocab as ner_vocab
from ner.scierc_dataloader import SCIERCDataset
from ner.vocab import (
    Vocab,
)
from ner.models import (
    bilstm_crf,
    elmo_bilstm_crf,
    cached_embedder,
    cached_bilstm_crf,
)
from ner import conlldataloader
from ner.models.cached_bilstm_crf import (
    CachedBiLSTMCRF,
)

from ner.models.cached_embedder import (
    CachedEmbedder,
)

from ner.models.elmo import (
    FrozenELMo,
)

from ner.utils import compute_f1_dataloader as ner_compute_f1

# local imports
from configurations.configuration import Configuration

DEFAULT_AL_CONFIG = Configuration('active_learning_manager_configuration.json')

def load_cached_embeder(cached_embedder: CachedEmbedder, session_dir: str) -> bool:
    path = os.path.join(session_dir, "cached_embedder.pkl")
    if os.path.exists(path):
        print("loading cached embedding vectors")
        with open(os.path.join(path), 'rb') as f:
            save_state = pickle.load(f)
            cached_embedder.load(save_state, 'cached_embedder')
    else:
        raise Exception("Could not find file: {}".format(path))

def load_model(vocab: Vocab, tag_vocab: Vocab, file_name: str, session_dir: str) -> nn.Module:
    if not os.path.exists(file_name):
        print(f'{file_name} does not exists')
        return None # skip not valid

    cached_embedder = CachedEmbedder(
        embedder=FrozenELMo.instance(),
        embedding_dimensions=FrozenELMo.DIMENSIONS,
    )
    load_cached_embeder(cached_embedder, session_dir)
    model = CachedBiLSTMCRF(
        vocab=vocab,
        tag_set=tag_vocab,
        hidden_dim=DEFAULT_AL_CONFIG.get_key('model_schema/hidden_dim'),
        batch_size=1,
        embedder=cached_embedder,
    )
    model.load_state_dict(torch.load(file_name))

    return model

def get_users(session_dir: str) -> List[str]:
    file_names = os.listdir(session_dir)
    users = []
    for file_name in file_names:
        full_path = os.path.join(session_dir, file_name)
        if os.path.isdir(full_path):
            users.append(file_name)
    return users

def get_data_loaders(session_dir: str, user_name: str):
    train_dataset = SCIERCDataset(os.path.join(session_dir, user_name, "train.txt"))
    valid_dataset = SCIERCDataset(os.path.join(session_dir, user_name, "test.txt"))

    train_dataset.parse_file()
    valid_dataset.parse_file()

    return train_dataset, valid_dataset

def compute_f1_data(model: torch.nn.Module, data_loader: object, tag_vocab: ner.vocab.Vocab) -> dict:
    return ner_compute_f1(
        model=model,
        data_loader=data_loader,
        tag_vocab=tag_vocab,
    )

def load_session_data(session_dir: str, ner_class: str):
    tag_vocab = ner_vocab.build_output_vocab([f'B-{ner_class}', f'I-{ner_class}', 'O'])
    with open(os.path.join(session_dir, "vocab.pkl"), 'rb') as f:
        vocab = pickle.load(f)
    with open(os.path.join(session_dir, "entry_to_sentences.pkl"), 'rb') as f:
        entry_to_sentences = pickle.load(f)
    with open(os.path.join(session_dir, "database.pkl"), 'rb') as f:
        database = pickle.load(f)
    
    users = get_users(session_dir)

    gold_data = SCIERCDataset(os.path.join(session_dir, "gold_set.txt"))
    gold_data.parse_file()

    user_data = {}

    with open('output.csv', 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')

        row_header = [
            'user_name',
            'train_f1', 'train_prec', 'train_rec', 'train_acc', # train metrics
            'valid_f1', 'valid_prec', 'valid_rec', 'valid_acc', # valid metrics
            'gold_f1', 'gold_prec', 'gold_rec', 'gold_acc', # gold metrics
        ]

        csv_writer.writerow(row_header)

        for user_name in users:
            model = load_model(
                vocab=vocab,
                tag_vocab=tag_vocab,
                file_name=os.path.join(session_dir, user_name, "model.ckpt"),
                session_dir=session_dir,
            )

            if model is None:
                continue

            train_dataset, valid_dataset = get_data_loaders(session_dir=session_dir, user_name=user_name)

            row_data = []
            for dataset, dataset_name in zip([train_dataset, valid_dataset, gold_data], ['train', 'valid', 'gold']):
                data_loader = conlldataloader.get_data_loader(
                    vocab,
                    tag_vocab,
                    dataset,
                    1,
                    False,
                    0,
                )

                f1_data, acc = compute_f1_data(model, data_loader, tag_vocab)
                user_data[user_name] = {
                    'model': model,
                    'f1_data': f1_data,
                    'acc': acc,
                }

                row_data.extend([
                    f1_data[ner_class]['f1'],
                    f1_data[ner_class]['precision'],
                    f1_data[ner_class]['recall'],
                    acc,
                ])

            csv_writer.writerow(row_data)

            ner.utils.print_f1_summary(f1_data)

    return user_data

def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='analyze user models and labels')
    parser.add_argument('--session_dir', required=True, help='the directory of the session to analyze')
    parser.add_argument('--ner_class', required=True, help='the class to analyze')
    return parser

def main():
    args = get_args().parse_args()
    user_data = load_session_data(args.session_dir, args.ner_class)

    return 0

if __name__ == "__main__":
    main()

