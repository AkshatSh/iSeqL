import unittest
import torch
from torch import nn

from ner.active_heuristic import (
    Random,
    Uncertantiy,
    KNNEmbeddings,
)

from . import utils

class TestActiveHeuristics(unittest.TestCase):
    '''
    Test cases for active_heuristics
    '''
    
    def test_random(self):
        '''
        run test cases for random heurisitcs
        '''
        # should not rely on vocabulary
        heuristic = Random(vocab=None, tag_vocab=None)
        dataset = utils.construct_sample_unlabeled_dataset()

        # this is model independent, so setting the model
        # to none should not change the output
        result = heuristic.evaluate(
            model=None,
            dataset=dataset,
        )

        model_result = heuristic.evaluate(
            model=utils.MockModel(),
            dataset=dataset,
        )

        # result should be model agnostic
        assert torch.all(result.eq(model_result))

        # make sure all the results are the same
        # postivie numbers
        # between 0 and 1
        # and sum to 1
        assert len(result) == len(dataset)
        for i in range(1, len(result)):
            assert result[i] >= 0.0
            assert result[i] <= 1.0
            assert result[i] == result[i - 1]
    
        assert sum(result) == 1.0
    
    def test_uncertain(self):
        '''
        test cases for uncertain based heuristic
        '''
        model = utils.MockModel()
        dataset = utils.construct_sample_unlabeled_dataset()
        vocab = utils.build_sample_vocab()
        tag_vocab = utils.build_sample_tag_vocab()

        heuristic = Uncertantiy(vocab=vocab, tag_vocab=tag_vocab)
        result = heuristic.evaluate(
            model=model,
            dataset=dataset
        )

        assert len(result) == len(dataset)
        # all result here should be equal to model.random_val
        assert torch.all(torch.eq(result, 1.0 / len(result)))
    
    def test_knn(self):
        '''
        test case for KNN based heuristic
        '''
        model = utils.MockModel()
        dataset = utils.construct_sample_unlabeled_dataset()
        vocab = utils.build_sample_vocab()
        tag_vocab = utils.build_sample_tag_vocab()

        heuristic = KNNEmbeddings(vocab=vocab, tag_vocab=tag_vocab)

        heuristic.prepare(
            model,
            dataset,
        )

        dataset.remove((0, utils.SAMPLE_DATASET[0][0]))

        output = heuristic.evaluate_with_labeled(
            model=model,
            dataset=dataset,
            labeled_indexes=[0],
            labeled_points=[(0,) + utils.SAMPLE_DATASET[0]],
        )

        assert len(output) == len(dataset)
        print(output)


