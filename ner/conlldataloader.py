import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
from tqdm import tqdm
from allennlp.modules.elmo import batch_to_ids

try:
    import constants
    from vocab import Vocab
except:
    from . import constants
    from .vocab import Vocab

class ConllDataSet(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = []
        self.word_list = []
        self.categories = []

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def parse_file(self):
        with open(self.file_name) as f:
            currInput = []
            currOutput = []
            for _, line in enumerate(tqdm(f)):
                if len(line.strip()) == 0:
                    # marks the end of a sentence
                    self.data.append(
                        {
                            'input': currInput,
                            'output': currOutput,
                        }
                    )
                    currInput = []
                    currOutput = []
                else:
                    tokens = line.split()
                    # seperates each line to 4 different things
                    # [word, pos, synchunk, output]
                    word, pos, synchunk, output = tokens
                    self.word_list.append(word)
                    currInput.append(
                        {
                            constants.CONLL2003_WORD : word,
                            constants.CONLL2003_POS: pos,
                            constants.CONLL2003_SYNCHUNK: synchunk,
                        }
                    )

                    if output not in self.categories:
                        self.categories.append(output)
                    
                    currOutput.append(output)

def default_conll_constructor(dataset: object):
    ''' 
    Create a dataset of (id, example) from a conll dataset
    '''
    return [
        (
            index,
            [inp[constants.CONLL2003_WORD] for inp in data_point['input']],
        )
        for index, data_point in enumerate(dataset.data)
    ]


class ConllDataSetUnlabeled(object):
    '''
    A conll data loader that only uses the input and output as sentences
    '''
    def __init__(
        self, 
        dataset: ConllDataSet,
        data_constructor: callable = default_conll_constructor,
    ):
        # self.file_name = dataset.filename
        # self.word_list = dataset.word_list
        # self.categories = dataset.categories
        self.data = data_constructor(dataset)

        self.ind2data = {
            index: data_in
            for index,data_in in self.data
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int) -> tuple:
        return self.data[index]
    
    def remove(self, inp):
        index, sentence = inp
        del self.ind2data[index]
        for i in range(len(self.data)):
            ind, string = self.data[i]
            if ind == index:
                del self.data[i]
                break

def unlabeled_example_fn(dataset: list, index: int) -> tuple:
    return dataset[index][0], dataset[index][1]

class ConllTorchDatasetUnlabeld(data.Dataset):
    def __init__(
        self, 
        vocab: Vocab, 
        categories: Vocab, 
        conlldataset: ConllDataSetUnlabeled,
        example_fn: callable = unlabeled_example_fn,
    ):
        '''
        Args:
            vocab: a vocab object to convert each token to a one hot encoding
            conlldataset: is an object containing all the input output pairs for the conlldataset
        '''
        self.vocab = vocab
        self.conll = conlldataset
        self.categories = categories
        self.example_fn = example_fn

    def __getitem__(self, index):
        s_id, sentence = self.example_fn(self.conll, index)
        encoded_sentence = [self.vocab(word) for word in sentence]

        sentence_tensor = torch.LongTensor(encoded_sentence)
        character_encoding = batch_to_ids([sentence])[0]

        # one hot encoding of the input (sentence length x vocab size)
        # one_hot_input = torch.zeros((sentence_tensor.shape[0], len(self.vocab)))
        # one_hot_input[torch.arange(sentence_tensor.shape[0]), sentence_tensor] = 1

        return s_id, sentence_tensor, character_encoding
    
    def __len__(self):
        return len(self.conll)


def collate_unlabeld_fn_with_sid(data):
    '''
    Args:
        data is a list of tuples of (sentence_tensor, one_hot_input, output_tensor, one_hot_output)
            one_hot_input and one_hot_output are the same dimensions
    '''
    data.sort(key=lambda x: len(x[1]), reverse=True)
    s_ids, inputs, character_encoding = zip(*data) 

    tensor_ids = torch.Tensor(s_ids).long()

    '''
    s_ids: (batch_size)
    inputs: (batch_size x sentence_length)
    outputs: (batch_size x sentence_length)
    '''

    lengths = [len(inp) for inp in inputs]
    
    batch_size = len(data)
    max_lengths = max(lengths) # max lengths

    batched_inputs = torch.zeros((batch_size, max_lengths))
    
    for i, inp in enumerate(inputs):
        end = lengths[i]
        batched_inputs[i, :end] = inp[:end]
    
    batched_characters = torch.zeros(
        (
            batch_size,
            max_lengths,
            character_encoding[0].shape[1] if len(character_encoding[0].shape) > 1 else 0,
        )
    ).long()

    for i, c_embed in enumerate(character_encoding):
        end = lengths[i]
        batched_characters[i, :end, :] = c_embed[:end, :]
    
    return tensor_ids, batched_inputs, batched_characters

def collate_unlabeled_fn(data):
    '''
    Args:
        data is a list of tuples of (sentence_tensor, one_hot_input, output_tensor, one_hot_output)
            one_hot_input and one_hot_output are the same dimensions
    '''
    _, batched_inputs, batched_characters = collate_unlabeld_fn_with_sid(data)
    return batched_inputs, batched_characters


def default_label_fn(conll: object, index: int):
    data_point = conll.data[index]
    data_input = data_point['input']
    data_output = data_point['output']
    sentence = [inp[constants.CONLL2003_WORD] for inp in data_input]
    return index, sentence, data_output

class ConllTorchDataset(data.Dataset):
    def __init__(self, vocab, categories, conlldataset, use_encoding=True, label_fn=default_label_fn, weight_fn:callable=None):
        '''
        Args:
            vocab: a vocab object to convert each token to a one hot encoding
            conlldataset: is an object containing all the input output pairs for the conlldataset
            use_encoding: returns the encoding of the word if true (one_hot tensor), if false returns the word
            itself. 
        '''
        self.vocab = vocab
        self.conll = conlldataset
        self.categories = categories
        self.use_encoding = use_encoding
        self.label_fn = label_fn
        self.weight_fn = weight_fn
    
    def __getitem__(self, index):
        sentence_id, sentence, data_output = self.label_fn(self.conll, index)

        if self.weight_fn is not None:
            weight = self.weight_fn(self.conll, index)
        else:
            weight = 1.0

        encoded_sentence = []
        encoded_sentence.extend([self.vocab(word) for word in sentence])

        # character encoding
        character_encoding = batch_to_ids([sentence])[0]

        encoded_output = []

        # start and end tokens don't have NER categories so they are the empty ones
        # 'O'
        encoded_output.extend([self.categories(token) for token in data_output])

        output_tensor = torch.LongTensor(encoded_output)

        # one hot encoding of output categories (setence length x categories size)
        one_hot = torch.zeros((output_tensor.shape[0], len(self.categories))) # one hot encoded representation of output

        one_hot[torch.arange(output_tensor.shape[0]), output_tensor] = 1
        sentence_tensor = torch.LongTensor(encoded_sentence)

        # one hot encoding of the input (sentence length x vocab size)
        one_hot_input = torch.zeros((sentence_tensor.shape[0], len(self.vocab)))
        one_hot_input[torch.arange(sentence_tensor.shape[0]), sentence_tensor] = 1

        return torch.LongTensor([sentence_id]), sentence_tensor, character_encoding, output_tensor, torch.Tensor([weight])
    
    def __len__(self):
        return len(self.conll)

def collate_fn_one_hot(data):
    '''
    Args:
        data is a list of tuples of (sentence, one_hot_input, data_output, one_hot_output)
            one_hot_input and one_hot_output are the same dimensions
    '''
    data.sort(key=lambda x: len(x[0]), reverse=True)
    _, inputs, _, outputs, _, = zip(*data) 

    lengths = [len(inp) for inp in inputs]
    
    batch_size = len(data)
    max_vocab = max([torch.max(inp) for inp in inputs]) #inputs[0].shape[1] # vocab
    max_lengths = max(lengths) # max lengths
    categories_size = max([torch.max(inp) for inp in outputs]) # categories size

    batched_inputs = torch.zeros((batch_size, max_lengths, max_vocab))
    
    # for i, inp in enumerate(inputs):
    #     end = lengths[i]
    #     batched_inputs[i, :end, :] = inp[i, :end, :]
    
    batched_outputs = torch.zeros((batch_size, max_lengths, categories_size))
    
    # for i, out in enumerate(outputs):
    #     end = lengths[i]
    #     batched_inputs[i, :end, :] = out[i, :end, :]
    

    return batched_inputs, batched_outputs

def collate_fn(data):
    '''
    Args:
        data is a list of tuples of (sentence_tensor, one_hot_input, output_tensor, one_hot_output)
            one_hot_input and one_hot_output are the same dimensions
    '''
    data.sort(key=lambda x: len(x[0]), reverse=True)
    s_ids, inputs, character_encoding, outputs, weights, = zip(*data) 

    '''
    inputs: (batch_size x sentence_length)
    outputs: (batch_size x sentence_length)
    '''

    lengths = [len(inp) for inp in inputs]
    
    batch_size = len(data)
    max_lengths = max(lengths) # max lengths

    batched_inputs = torch.zeros((batch_size, max_lengths))
    
    for i, inp in enumerate(inputs):
        end = lengths[i]
        batched_inputs[i, :end] = inp[:end]
    
    batched_outputs = torch.zeros((batch_size, max_lengths))
    
    for i, out in enumerate(outputs):
        end = lengths[i]
        batched_outputs[i, :end] = out[:end]
    
    batched_characters = torch.zeros(
        (
            batch_size,
            max_lengths,
            character_encoding[0].shape[1],
        )
    ).long()

    for i, c_embed in enumerate(character_encoding):
        end = lengths[i]
        batched_characters[i, :end, :] = c_embed[:end, :]
    
    s_ids_tensor = torch.zeros((batch_size, 1)).long()
    for i, s_id in enumerate(s_ids):
        s_ids_tensor[i] = s_id
    
    weight_tensor = torch.zeros((batch_size, 1))
    for i, weight in enumerate(weights):
        weight_tensor[i] = weight
    return s_ids_tensor, batched_inputs,  batched_characters, batched_outputs, weight_tensor


def get_data_loader(vocab, categories, conlldataset, batch_size, shuffle, num_workers, one_hot=False, label_fn=default_label_fn, weight_fn=None):
    # set up torch dataset
    conll_dataset = ConllTorchDataset(vocab, categories, conlldataset, label_fn=label_fn, weight_fn=weight_fn)

    data_loader = data.DataLoader(
        dataset=conll_dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=(collate_fn_one_hot if one_hot else collate_fn)
    )
    
    return data_loader

def get_unlabeled_data_loader(
    vocab: Vocab,
    categories: Vocab,
    unlabeled_data: ConllDataSetUnlabeled,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    unlabeled_example_fn=unlabeled_example_fn,
    collate_fn=collate_unlabeled_fn,
):
    conll_dataset = ConllTorchDatasetUnlabeld(
        vocab,
        categories,
        unlabeled_data,
        example_fn=unlabeled_example_fn,
    )

    data_loader = data.DataLoader(
        dataset=conll_dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return data_loader
