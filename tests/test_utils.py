import unittest
import os
from ner.utils import (
    sample_array,
    get_or_create_file,
    compute_avg_f1,
    compute_f1_dataloader,
    compute_labels,
    get_simple_entity,
)

from ner.conlldataloader import (
    ConllTorchDatasetUnlabeld,
    get_unlabeled_data_loader,
)

from . import utils

class TestUtils(unittest.TestCase):

    def test_compute_labels(self):
        result = compute_labels(
            model=utils.build_all_models()[1],
            data_loader=get_unlabeled_data_loader(
                vocab=utils.build_sample_vocab(),
                categories=utils.build_sample_tag_vocab(),
                unlabeled_data=utils.construct_sample_unlabeled_dataset(),
                batch_size=1,
                shuffle=False,
                num_workers=0,
            ),
            tag_vocab=utils.build_sample_tag_vocab(),
            verbose=True,
        )

        assert len(result) == len(utils.construct_sample_unlabeled_dataset())

    def test_empty_compute_avg_f1(self):
        '''
        Tests that this should not break for empty f1 information
        '''
        result = compute_avg_f1({})
        assert result == 0
    
    def test_compute_f1_dataloader_empty(self):
        '''
        Tests that this should not break for empty datasets
        '''
        res = compute_f1_dataloader(
            utils.build_all_models()[0],
            [], # empty dataset
            utils.build_sample_tag_vocab(),
        )

    def _test_sample(self, array_len: int, percent: float):
        '''
        tests creating a sample array, of length array_len, 
        and creates a random `percent` sample of the array

        such that the size of the new array is int(sample_size * percent)
        and all the elements are randomly sampled from the sample array
        '''
        random_arr = [ i for i in range(array_len)]

        arr = sample_array(random_arr, percent)
        
        assert len(arr) == int(percent * array_len)

        # make sure sample array is subset of random_arr
        for item in arr:
            assert item in random_arr

    def test_sample_array(self):
        '''
        Try out sampling an array of various sizes
        '''
        self._test_sample(100, 0.1)
        self._test_sample(200, 0.5)
        self._test_sample(300, 1.0)
    
    def test_get_or_create_file(self):
        '''
        test creation of a file and retrieval and ensure file
        updates properly
        '''
        file_data = [
            'temp\n',
            'temp2\n'
        ]
        file_name = 'test_rand/temp.txt'

        # should create
        new_file = get_or_create_file(file_name)
        new_file.write(file_data[0])
        new_file.close()
        assert os.path.exists(file_name)

        # should not create
        same_file = get_or_create_file(file_name)
        same_file.write(file_data[1])
        same_file.close()

        # should have appended both cases to each other
        with open(file_name) as f:
            for i, line in enumerate(f):
                # should only have last line (it should overwrite)
                assert file_data[i+ 1].strip() == line.strip()

        # clean up
        os.remove(file_name) 
        os.rmdir('test_rand/')
    
    def test_simple_entity(self):
        '''
        A utility test for get simple entity which converts a sequence of words
        to its basic form by stripping stop words, removing punctuation and 
        returning the lemma for each word
        '''
        res = get_simple_entity('here is a')
        assert type(res) == list
        assert len(res) == 0
        res = get_simple_entity('quite painful legs')
        assert len(res) == 2
        assert type(res) == list
        self.assertEqual(res, ['leg', 'painful'])
        res_str = get_simple_entity('quite painful legs', True)
        assert type(res_str) == str
        self.assertEqual(res_str, ' '.join(res))
        res_str = get_simple_entity('I have painful legs', True)
