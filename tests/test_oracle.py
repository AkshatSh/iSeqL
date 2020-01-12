import unittest
from typing import (
    List,
    Tuple,
)

from ner import constants

from ner.oracle import (
    SimulatedOracle,
)

from ner.conlldataloader import (
    ConllDataSet,
)

from . import utils

class TestOracle(unittest.TestCase):

    def test_simluated_retrevial(self):
        '''
        Make sure that the simlated oracle is able to answer
        queries by checking against the ground truth dataset
        '''
        dataset = utils.SAMPLE_DATASET
        dataloader = utils.construct_dataloader(dataset)
        oracle = SimulatedOracle(dataloader)

        for i, (input_point, output_point) in enumerate(dataset):
            query = (i, input_point)
            s_id, sent, result = oracle.get_label(query)

            # tests
            assert sent == input_point
            assert len(result) == len(input_point)
            assert output_point == result

