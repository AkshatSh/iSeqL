from typing import (
    List,
    Tuple,
    Dict,
    Callable,
)

import pickle
import argparse
import os
from tqdm import trange, tqdm
import torch
import copy

from tensor_logger import Logger
import utils
import conlldataloader
from vocab import Vocab, BinaryVocab
import oracle
import active_heuristic
import constants
from ner import weak_args
import conll_utils
import dataset_utils

import models.bilstm_crf as bilstm_crf
from models.advanced_tutorial import BiLSTM_CRF
import models.elmo_bilstm_crf as elmo
import models.crf as crf
import models.dictionary_model as dict_model
import scierc_utils
from ner.trainer import Trainer

def build_weak_set(
    model: torch.nn.Module,
    unlabeled_dataset: conlldataloader.ConllDataSetUnlabeled,
    vocab: Vocab,
    tag_vocab: Vocab,
    device: str,
    weak_weight: float,
) -> List[Tuple[int, List[str], List[str]]]:
    data_loader = conlldataloader.ConllTorchDatasetUnlabeld(
        vocab,
        tag_vocab,
        unlabeled_dataset,
    )

    outputs = []
    for i, (s_ids, x, x_chars) in enumerate(tqdm(data_loader, total=len(data_loader))):
        x, x_chars, s_ids = x.to(device), x_chars.to(device), torch.Tensor([s_ids]).to(device)
        x = x.unsqueeze(0)
        x_chars = x_chars.unsqueeze(0)
        s_ids = s_ids.unsqueeze(0)
        model_out = model(x, x_chars, s_ids)
        predicted = [tag_vocab.get_word(int(index)) for index in model_out[0]]
        outputs.append(predicted)

    weak_set = []

    for (s_id, sentence), prediction in zip(unlabeled_dataset, outputs):
        weak_set.append(
            (s_id, sentence, prediction, weak_weight)
        )

    return weak_set


def active_train(
    log_dir: str,
    model: torch.nn.Module,
    model_path: str,
    unlabeled_dataset: conlldataloader.ConllDataSetUnlabeled,
    test_dataset: conlldataloader.ConllDataSet,

    # active learning parameters
    iterations: int,
    heuritic: active_heuristic.ActiveHeuristic,
    oracle: oracle.Oracle,
    sample_size: int,
    sampling_strategy: str,  # sampling, top_k

    # train parameters
    vocab: Vocab,
    tag_vocab: Vocab,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    num_epochs: float,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
    optimizer_type: str,
    device: str,
    summary_file: str,
    dictionary_model: torch.nn.Module,
    weak_weight: float,
) -> None:
    logger = Logger(
        os.path.join(
            log_dir, 
            "{}/".format(model_path)
        ),
        summary_file=summary_file,
    )
    
    # random sample dataset into
    train_data = []

    test_data_loader = conlldataloader.get_data_loader(
        vocab,
        tag_vocab,
        test_dataset,
        1,  # batch_size
        False, # no shuffle
        1, # 1 worker
    )

    start_model = copy.deepcopy(model)

    iteration_samples = [1, 5, 10, 25, 50, 100, 200, 400, 400]

    labeled_indexes = []

    for i, sample_size in enumerate(iteration_samples):
        # select new points from distribution
        if isinstance(heuritic, active_heuristic.KNNEmbeddings):
            distribution = heuritic.evaluate_with_labeled(
                model=model,
                dataset=unlabeled_dataset,
                labeled_indexes=labeled_indexes,
                labeled_points=train_data,
                device=device
            )
        else:
            distribution = heuritic.evaluate(model, unlabeled_dataset, device)
        new_points = []
        sample_size = min(sample_size, len(distribution) - 1)
        if sampling_strategy == constants.ACTIVE_LEARNING_SAMPLE:
            new_points = torch.multinomial(distribution, sample_size)
        elif sampling_strategy == constants.ACTIVE_LEARNING_TOP_K:
            new_points = sorted(
                range(len(distribution)), 
                reverse=True,
                key=lambda ind: distribution[ind]
            )
        new_points = new_points[:sample_size]


        # use new points to augment train_dataset
        # remove points from unlabaled corpus
        query = [
            unlabeled_dataset.data[ind]
            for ind in new_points
        ]

        labeled_indexes.extend(
            ind for (ind, _) in query
        )

        # adds a weight
        outputs = [oracle.get_label(q) + (1.0,) for q in query]

        # move unlabeled points to labeled points
        [unlabeled_dataset.remove(q) for q in query]

        train_data.extend(outputs)

        dict_trainer = Trainer(
            model=dictionary_model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            optimizer_type=optimizer_type,
            vocab=vocab,
            tags=tag_vocab,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            train_dataset=train_data,
            test_dataset=test_dataset,
            logger=logger,
            device=device,
            verbose_print=True,
            verbose_log=True,
            train_label_fn=lambda data, index : (data[index][:3]),
            train_weight_fn=lambda data, index: (data[index][3])
        )

        dict_trainer.train(1)
        dictionary_model = dict_trainer.get_best_model()

        weak_set = build_weak_set(
            model=dictionary_model,
            unlabeled_dataset=unlabeled_dataset,
            vocab=vocab,
            tag_vocab=tag_vocab,
            device=device,
            weak_weight=weak_weight,
        )


        trainer = Trainer(
            model=copy.deepcopy(model),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            optimizer_type=optimizer_type,
            vocab=vocab,
            tags=tag_vocab,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            train_dataset=(train_data + weak_set),
            test_dataset=test_dataset,
            logger=logger,
            device=device,
            verbose_print=True,
            verbose_log=True,
            train_label_fn=lambda data, index : (data[index][:3]),
            train_weight_fn=lambda data, index: (data[index][3])
        )

        trainer.train(num_epochs)
        model = trainer.get_best_model()



        # compute valid metrics
        f1_data, acc = utils.compute_f1_dataloader(model, test_data_loader, tag_vocab, device=device)
        f1_avg_valid = utils.compute_avg_f1(f1_data)

        # log valid metics
        logger.scalar_summary("active valid f1", f1_avg_valid, len(train_data))
        logger.scalar_summary("active valid accuracy", acc, len(train_data))
        utils.log_metrics(logger, f1_data, "active valid", len(train_data))

        print(f'Finished experiment on training set size: {len(train_data)}')

    logger.flush()

def main():
    args = weak_args.get_arg_parser().parse_args()

    # determine device
    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    print("using device {} ...".format(device))

    model_type = 'bilstm_crf' if args.train_bi_lstm else 'elmo_bilstm_crf'
    model_type = 'dictionary' if args.train_dictionary else model_type
    model_type = 'cached' if args.train_cached else model_type
    model_type = 'phrase_dictionary' if args.train_phrase_dictionary else model_type
    out = dataset_utils.load_dataset(args, force_load=True)
    train_dataset, valid_dataset, train_vocab, output_categories = out

    if args.binary_classifier:
        b_class = args.binary_classifier
        print('converting to a binary problem for class: {}'.format(b_class))
        output_categories = BinaryVocab(output_categories, select_class=b_class)

    # phrase: 69 F1 Drug 791 examples
    # phrase:  58 F1 ADR 791 examples
    # word: 69 F1 Drug 791 examples
    # word: 59 F1 ADR 791 examples

    # build unlabeled corpus
    unlabeled_corpus = conlldataloader.ConllDataSetUnlabeled(train_dataset)

    model = utils.build_model(
        model_type=model_type,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        vocab=train_vocab,
        tag_vocab=output_categories,
    ).to(device)

    if args.weak_dictionary == 'word':
        dictionary_model = utils.build_model(
            model_type='dictionary',
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            vocab=train_vocab,
            tag_vocab=output_categories,
        )
    elif args.weak_dictionary == 'phrase':
        dictionary_model = utils.build_model(
            model_type='phrase_dictionary',
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            vocab=train_vocab,
            tag_vocab=output_categories,
        )
    else:
        raise Exception(f'Unknown dictionary model type: {args.weak_dictionary}')

    if model_type == 'cached':
        model.embedder.cache_dataset(unlabeled_corpus, verbose=True, device=device)

    # created a simulated oracle with all the ground truth values
    sim_oracle = oracle.SimulatedOracle(train_dataset)

    # heuristic
    if args.heuristic == constants.ACTIVE_LEARNING_RANDOM_H:
        h = active_heuristic.Random(train_vocab, output_categories)
    elif args.heuristic == constants.ACTIVE_LEARNING_UNCERTAINTY_H:
        h = active_heuristic.Uncertantiy(train_vocab, output_categories)
    elif args.heuristic == constants.ACTIVE_LEARNING_KNN:
        h = active_heuristic.KNNEmbeddings(train_vocab, output_categories)
        h.prepare(
            model=model,
            dataset=unlabeled_corpus,
            device=device,
        )
    else:
        raise Exception("Unknown heurisitc: {}".format(args.heuristic))

    active_train(
        log_dir=args.log_dir,
        model=model,
        model_path=args.model_path,
        unlabeled_dataset=unlabeled_corpus,
        test_dataset=valid_dataset,

        # active learning parameters
        iterations=args.iterations,
        heuritic=h,
        oracle=sim_oracle,
        sample_size=args.sample_size,
        sampling_strategy=args.sampling_strategy,

        # train parameters
        vocab=train_vocab,
        tag_vocab=output_categories,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        optimizer_type=args.optimizer_type,

        # Other parameters
        device=device,
        summary_file=args.summary_file,

        dictionary_model=dictionary_model,
        weak_weight=args.weak_weight,
    )

if __name__ == "__main__":
    main()