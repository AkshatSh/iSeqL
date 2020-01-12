import torch
import os 
import pickle
import sys

from typing import (
    List,
    Tuple,
    Dict,
)

from ner.models import cached_embedder

from ner.active_heuristic import KNNEmbeddings
from ner.models.elmo import FrozenELMo

def construct_cached_embedder_from_knn(knn: KNNEmbeddings) -> cached_embedder.CachedEmbedder:
    index_map = knn.index_map
    database = knn.database
    sentence_start = knn.sentence_start
    sentence_end = knn.sentence_end
    embedding_dimension = FrozenELMo.DIMENSIONS


    ce = cached_embedder.CachedEmbedder(
        embedder=FrozenELMo.instance(),
        embedding_dimensions=embedding_dimension,
    )

    ce.load(
        {
            "cached_embedder": (
                index_map,
                database,
                sentence_start,
                sentence_end,
                embedding_dimension,
            )
        },
        "cached_embedder",
    )

    return ce

def is_positive_label(label: str) -> bool:
    '''
    Arguments:
        * label (str) the current label
    
    Returns:
        * (bool) whether the current label is in the 
        positive class
    '''
    return (
        len(label) > 2 and
        (
            label[0] == 'B' or
            label[0] == 'I'
        )
    )

def is_negative_label(label: str) -> bool:
    '''
    Arguments:
        * label (str) the current label
    
    Returns:
        * (bool) whether the current label is in the 
        negative class
    '''
    return len(label) > 0 and label[0] == 'O'

def compute_flip(prev_label: List[str], curr_label: List[str]) -> Tuple[int, int]:
    '''
    Given a previous label and a current label computes, how many examples
    are positive flipped and negative flipped

    where positive flipped is 'O' switching to a class (B-Class or I-Class)
    and negative flipped is (B-Class or I-Class) switching to 'O'.
    '''
    pos_flipped = 0
    neg_flipped = 0
    for prev, curr in zip(prev_label, curr_label):
        if is_positive_label(prev) and is_negative_label(curr):
            neg_flipped += 1
        elif is_negative_label(prev) and is_positive_label(curr):
            pos_flipped += 1
    return (pos_flipped, neg_flipped)

def compute_total_flip(
    prev_labels: List[List[str]],
    curr_labels: List[List[str]],
) -> Tuple[int, int]:
    '''
    Given a previous label and a current label computes, how many examples
    are positive flipped and negative flipped

    where positive flipped is 'O' switching to a class (B-Class or I-Class)
    and negative flipped is (B-Class or I-Class) switching to 'O'.

    Arguments:
        * prev_labels (List[List[str]]) all the labels for the previous iteration
        * curr_labels (List[List[str]]) all the labels from the current iteration
    
    Returns:
        Tuple[int, int]
            - first item is the number of positive flipped
            - second item is the number of negative flipped
    '''
    if prev_labels is None:
        # on first iteration nothing has flipped
        return (0, 0)
    
    pos_flip = 0
    neg_flip = 0
    for prev_label, curr_label in zip(prev_labels, curr_labels):
        curr_pos_flip, curr_neg_flip = compute_flip(prev_label, curr_label)
        pos_flip += curr_pos_flip
        neg_flip += curr_neg_flip
    return (pos_flip, neg_flip)

def conllize_database(
    dir_name: str,
    database: Dict[str, List[Tuple[str, str]]],
    train_ids: List[int],
    test_ids: List[int],
):
    '''
    Given a filename and a database serialize the filename and database
    into a conll type database
    '''
    train_file_name = os.path.join(dir_name, 'train.txt')
    test_file_name = os.path.join(dir_name, 'test.txt')
    total_database_file_name = os.path.join(dir_name, 'database.txt')
    valid_db_items = filter(lambda entry: entry[1][1] is not None, database.items())
    train_db_items = filter(lambda entry: entry[0] in train_ids, database.items())
    test_db_items = filter(lambda entry: entry[0] in test_ids, database.items())

    def _write_to_file(f, db_items):
        for _, (input_str, labels) in db_items:
            for input_w, label in zip(input_str, labels):
                f.writelines(f'{input_w}\t{label}\n')
            f.writelines('\n')

    with open(total_database_file_name, 'w+') as f:
        _write_to_file(f=f, db_items=valid_db_items)
    
    with open(train_file_name, 'w+') as f:
        _write_to_file(f=f, db_items=train_db_items)
    
    with open(test_file_name, 'w+') as f:
        _write_to_file(f=f, db_items=test_db_items)