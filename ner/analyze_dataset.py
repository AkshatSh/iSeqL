# Global imports
from typing import (
    List,
    Dict,
    Tuple,
)
import pickle
import argparse
import os
import torch
from tqdm import trange, tqdm

# local imports
import vocab
import constants
from train_args import get_arg_parser
from tensor_logger import Logger
import utils

# Modeling Imports
import models.bilstm_crf as bilstm_crf
from models.advanced_tutorial import BiLSTM_CRF
import models.elmo_bilstm_crf as elmo
import models.crf as crf
import models.dictionary_model as dict_model

import dataset_utils
import conlldataloader

from trainer import Trainer

def main(args):
    '''
    main method
    '''
    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    print("using device {} ...".format(device))

    out = dataset_utils.load_dataset(args)
    if args.load or args.save:
        train_dataset, valid_dataset, train_vocab, output_categories = out
        # train_vocab = vocab.unk_vocab(train_vocab)
    elif args.clean:
        return
    
    if args.binary_classifier:
        b_class = args.binary_classifier
        print('converting to a binary problem for class: {}'.format(b_class))
        output_categories = vocab.BinaryVocab(output_categories, select_class=b_class)

    total_tokens = 0
    total_class_tokens = 0
    
    def explain_labels(
        example: List[str],
        seq_label: List[str],
    ) -> Tuple[List[Tuple[int, int]], List[str]]:
        '''
        Convert a label to a list of word ranges and entities

        word_range[i] = (start: int, end: int) with end exclusive
        entities[i] = str, the entity corresponding to word_range[i]
        '''
        ranges : list = []
        entities: list = []
        range_start : int = None
        seq_label = [] if seq_label is None else seq_label
        for i, label in enumerate(seq_label):
            if (label == 'O' or i == len(seq_label) - 1) and range_start is not None:
                    ranges.append(
                        (
                            range_start,
                            i,
                        )
                    )
                    entities.append((example[range_start : i]))
                    range_start = None
            elif label.startswith('B'):
                if range_start is not None:
                    ranges.append(
                        (
                            range_start,
                            i,
                        )
                    )
                    entities.append((example[range_start : i]))
                range_start = i

        return ranges, entities
    # for item in train_dataset.data:
    #     sent = [inp['word'] for inp in item ]
    #     out = item['output']
    #     total_tokens += len(out)
    #     num_pos = 0
    #     for out_i in out:
    #         if len(out_i) > 0 and out_i[2:] == b_class:
    #             num_pos += 1
    #     total_class_tokens += num_pos 

    all_ents = []
    has_ents = []
    for item in train_dataset.data:
        sent = [inp['word'] for inp in item['input']]
        out = item['output']
        total_tokens += len(out)
        num_pos = 0
        act_out = []
        for out_i in out:
            if len(out_i) > 0 and out_i[2:] == b_class:
                num_pos += 1
                act_out.append(out_i)
            else:
                act_out.append('O')
        r, e=  explain_labels(sent, act_out)
        has_ents.append(len(e) > 0)
        all_ents.extend(e)
        total_class_tokens += num_pos

    per = sum(has_ents)/len(has_ents)
    print(f'has_ents: {len(has_ents)} ents: {sum(has_ents)} per: {per}')
    print(f'Num Ents: {len(all_ents)}')

    lens = [len(ent) for ent in all_ents]
    avg_size = sum(lens) / len(lens)
    print(f'Average Size: {avg_size}') 
    
    print(f'Positive Tokens: {total_class_tokens} | Total: {total_tokens} | Percent: {total_class_tokens / total_tokens}')

if __name__ == "__main__":
    main(
        get_arg_parser().parse_args(),
    )
