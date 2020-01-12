# -*- coding: utf-8 -*-
import sys
sys.path.append("..") 

import os
import h5py
import csv
import numpy as np
import argparse
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder
import nltk
import torch
import scipy
from tqdm import tqdm
from typing import List
import random
import utils

import scierc_utils
import conlldataloader
import conll_utils
import constants
import vocab
from models.elmo import FrozenELMo

def load_elmo(device: str) -> Elmo:
    '''
    load a pretrained elmo model
    '''
    model = FrozenELMo.instance()
    return model

def get_embedding(
    elmo_model: Elmo,
    elmo_embedder: ElmoEmbedder,
    device: str,
    vocab: vocab.Vocab,
    raw_sentence: List[str]
) -> torch.Tensor:
    embeded_sentence = elmo_model.get_embedding_from_sentence(raw_sentence[0]).unsqueeze(0)
    # embeded_sentence = elmo_embedder.batch_to_embeddings(raw_sentence)[0]
    # embeded_sentence *= torch.ones(embeded_sentence.shape)
    # embeded_sentence = embeded_sentence.sum(1)
    # character_ids = batch_to_ids(raw_sentence).to(device)
    # embeddings = elmo_model(character_ids)
    # embeded_sentence = embeddings['elmo_representations'][0]

    return embeded_sentence

def write_elmo(
    elmo: Elmo,
    elmo_embedder: ElmoEmbedder,
    device: str,
    dataset: conlldataloader.ConllDataSet,
    vocab: vocab.Vocab,
    tag_vocab: vocab.Vocab,
    folder: str,
) -> None:
    print('writing elmo embeddings ')

    dim = 1024
    p_out = os.path.join(folder, 'latent')
    if not os.path.exists(p_out):
        os.makedirs(p_out)
    p_h5 = os.path.join(p_out, 'latent{}.h5'.format(dim))
    p_meta = os.path.join(p_out, 'meta{}.csv'.format(dim))

    data_loader = conlldataloader.get_data_loader(
        vocab,
        tag_vocab,
        dataset,
        1,
        False,
        2, # num workers
    )

    embedding_space = {}

    for i, (sentence, sentence_chars, label) in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            sentences = []
            for k in range(len(sentence)):
                sentences.append(vocab.decode(sentence[k]))
            
            tags = []
            for k in range(len(label)):
                tags.append(tag_vocab.decode(label[k]))

            embedding = elmo(sentence_chars)['elmo_representations'][0]
            # embedding = get_embedding(elmo, elmo_embedder, device, vocab, sentences)
            for k, raw_sentence in enumerate(sentences):
                for j, word in enumerate(raw_sentence):
                    if (tags[k][j] != 'O' or random.random() < 0.1):
                        embedding_space[
                            (
                                word,
                                ' '.join(raw_sentence),
                            )
                        ] = (
                            np.array(embedding[k][j]),
                            tags[k][j],
                        )


    print('Built embedding space with {} entries'.format(len(embedding_space)))

    res = []
    
    with open(p_meta, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['i', 'name', 'tag', 'context'])
        for i, (word, context) in enumerate(tqdm(embedding_space)):
            embedding, tag = embedding_space[(word, context)]
            res.append(embedding)
            removed_bio_tag = utils.remove_bio_input(tag)
            writer.writerow([i, word, removed_bio_tag, "\"{}\"".format(context)])

    f = h5py.File(p_h5, 'w')
    dset = f.create_dataset('latent', data=res)
    f.close()

    print('done writing elmo embeddings')

def get_arg_parser() -> argparse.ArgumentParser:
    '''
    Create arg parse for training containing options for
        optimizer
        hyper parameters
        model saving
        log directories
    '''
    parser = argparse.ArgumentParser(description='Train Named Entity Recognition on Train Conll2003.')

    # Parser data loader options
    parser.add_argument('--dataset', type=str, default='conll', help='use conll 2003 dataset')
    parser.add_argument('--output', type=str, default="out/", help='the output folder')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')

    return parser

def main() -> None:
    args = get_arg_parser().parse_args()
    if args.dataset == 'conll':
        train_dataset, valid_dataset, train_vocab, output_categories = conll_utils.load()
    elif args.dataset == 'SCIERC':
        train_dataset, valid_dataset, train_vocab, output_categories = scierc_utils.load()
    else:
        raise Exception("Unknown dataset: {}".format(args.dataset))
    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    print('Using device: {}'.format(device))
    elmo = load_elmo(device)

    write_elmo(elmo, ElmoEmbedder(), device, train_dataset, train_vocab, output_categories, args.output)

if __name__ == '__main__':
    main()
