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

def train_bilstm_crf(
    train_dataset,
    test_dataset,
    vocab,
    tag_vocab,
    batch_size,
    shuffle,
    num_workers,
    num_epochs,
    embedding_dim,
    hidden_dim,
    learning_rate,
    weight_decay,
    momentum,
    optimizer_type,
    log_dir,
    save_dir,
    model_name,
    model_path,
    sample,
    summary_file: str,
    training_threshold: int,
    model_type="bilstm_crf",
    device: str = 'cpu',
):
    if not os.path.exists(save_dir):
        # create the save directory for the model
        os.makedirs(save_dir)
    
    # if not os.path.exists(os.path.join(save_dir, model_path, model_name)):
    #     # create the save directory specifically for the model
    #     os.makedirs(os.path.join(save_dir, model_path, model_name))
    
    if sample < 1:
        print("Using {} % of the training data".format(int(sample * 100)))
        train_dataset.data = utils.sample_array(train_dataset.data, sample)
    
    test_data_loader = conlldataloader.get_data_loader(
        vocab,
        tag_vocab,
        test_dataset,
        1,  # batch_size
        False,
        1,
    )

    logger = Logger(
        os.path.join(
            log_dir, 
            "{}/".format(model_name)
        ),
        summary_file=summary_file,
    )

    model = utils.build_model(
        model_type=model_type,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        vocab=vocab,
        tag_vocab=tag_vocab,
    ).to(device)
    print(model)

    trainer = Trainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        optimizer_type=optimizer_type,
        vocab=vocab,
        tags=tag_vocab,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        logger=logger,
        device=device,
        verbose_print=True,
        verbose_log=True,
        threshold=training_threshold,
    )

    best_epoch, best_epoch_summary = trainer.train(num_epochs)

    logger.flush()
    with open('log.txt', 'a') as log_file:
        utils.analyze_predictions(
            trainer.get_best_model(),
            test_data_loader, 
            vocab,
            tag_vocab,
            log_file,
            device,
        )
    
    torch.save(
        trainer.get_best_model().state_dict(), 
        os.path.join(save_dir, "{}.ckpt".format(model_name)),
    )
    return


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

    model_type = None
    if args.train_bi_lstm:
        model_type = "bilstm_crf"
    elif args.train_elmo_bi_lstm:
        model_type = "elmo_bilstm_crf"
    elif args.train_dictionary:
        model_type = "dictionary"
    
    if args.train:
        train_bilstm_crf(
            train_dataset=train_dataset,
            test_dataset=valid_dataset,
            vocab=train_vocab,
            tag_vocab=output_categories,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            num_epochs=args.num_epochs,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            optimizer_type=args.optimizer_type,
            log_dir=args.log_dir,
            save_dir=args.save_dir,
            model_name=args.model_name,
            model_path=args.model_path,
            sample=args.sample,
            summary_file=args.summary_file,
            model_type=model_type,
            device=device,
            training_threshold=args.training_threshold,
        )

if __name__ == "__main__":
    main(
        get_arg_parser().parse_args(),
    )
