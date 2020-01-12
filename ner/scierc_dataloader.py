import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
from tqdm import tqdm

try: # pragma: no cover
    import constants
    from vocab import Vocab
except: # pragma: no cover
    from . import constants
    from .vocab import Vocab

class SCIERCDataset(object):
    '''
    Given the directory of the SCIERC dataset, parses the all the files and
    creates a data list, where the list is tuples of (input, tagged) where input and tagged
    are two lists of the exact same elements, and tagged contains all the BIO tags for
    the input.

    Arguments:
        file_name: the name of the directory that contains all data from SCIERC in a conllized format
    '''
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.data = []
        self.word_list = []
        self.categories = []

    def __len__(self) -> int:
        return len(self.data)
    
    def parse_file(self) -> None:
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
                    word, output = tokens
                    self.word_list.append(word)
                    currInput.append(
                        {
                            constants.CONLL2003_WORD : word,
                        }
                    )

                    if output not in self.categories:
                        self.categories.append(output)
                    
                    currOutput.append(output)