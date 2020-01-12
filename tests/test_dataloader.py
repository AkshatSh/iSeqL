import unittest
import copy

from ner import conlldataloader, constants

from . import utils

class TestDataloader(unittest.TestCase):

    def test_remove_unlabeled(self):
        dataset = utils.construct_sample_unlabeled_dataset()

        old_len = len(dataset)

        # in sample dataset index 1 is unique
        to_remove = dataset.data[1]
        dataset.remove(to_remove)

        # only one item was deleted
        assert len(dataset) == old_len - 1

        for index, (inp_i, inp_entry) in enumerate(dataset):
            # make sure entries do not exist in the dataset anymore
            assert not inp_i == to_remove[0]
            assert not inp_entry == to_remove[1]

    def test_iteration(self):
        '''
        test the iteration of dataloaders to make sure dataloaders 
        produce the expected shape
        '''
        torch_dataset = utils.build_sample_torch_dataset()

        for i, entry in enumerate(torch_dataset):

            # make sure there are five things in each entry
            assert len(entry) == 5

            # assert the shapes match up where need be
            s_ids, sentence_tensor, character_encoding, output_tensor, weight = entry

            # shape 0 is the length of the sentence in unbatched
            # cases
            assert sentence_tensor.shape[0] == output_tensor.shape[0]
            assert sentence_tensor.shape[0] == character_encoding.shape[0]
            assert weight.shape[0] == s_ids.shape[0]
    
    def test_parse_conll(self):
        '''
        Ensure that the file parser is able to parse the file for CONLL
        with no errors
        '''
        train_dataset = conlldataloader.ConllDataSet(constants.CONLL2003_TRAIN)
        train_dataset.parse_file()

    def test_conll_dataset_getter(self):
        '''
        verifies __getitem__ of ConllDataSet is working
        '''
        ds = utils.construct_sample_dataloader()
        item = ds[0]
        assert item is not None

    def test_torch_unlabeled_iteration(self):
        '''
        A test case to ensure unlabeled dataset iterates and collates with
        no problem
        '''
        # Data
        vocab = utils.build_sample_vocab()
        tag_vocab = utils.build_sample_tag_vocab()
        unlabeled_dataset = utils.construct_sample_unlabeled_dataset()

        # Parameters
        batch_size = 2
        shuffle = False
        num_works = 0

        data_loader = conlldataloader.get_unlabeled_data_loader(
            vocab=vocab,
            categories=tag_vocab,
            unlabeled_data=unlabeled_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_works,
        )

        for i, entry in enumerate(data_loader):
            # ensure iteration works
            pass
    
    def test_one_hot_iteration(self):
        '''
        A test to ensure one hot iteration occurs with no problem
        '''
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
            one_hot=True
        )

        for i, entry in enumerate(data_loader):
            # ensure iteration works
            pass
