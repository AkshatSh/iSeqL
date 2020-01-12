import os
import sys
import pickle
from typing import List, Tuple, Dict
import json
import pprint
import nltk
from tqdm import tqdm
import random

from ner.constants import (
    # imports of raw data
    SCIERC_PROCESSED_TRAIN,
    SCIERC_PROCESSED_VALID,
    SCIERC_PROCESSED_TEST,

    # imports of locations
    SCIERC_CONLL_PROCESSED_TRAIN_DATASET,
    SCIERC_CONLL_PROCESSED_VALID_DATASET,
    SCIERC_CONLL_PROCESSED_TEST_DATASET,
)

SCIERC_FILES = [
    (SCIERC_PROCESSED_TRAIN, SCIERC_CONLL_PROCESSED_TRAIN_DATASET),
    (SCIERC_PROCESSED_VALID, SCIERC_CONLL_PROCESSED_VALID_DATASET),
    (SCIERC_PROCESSED_TEST, SCIERC_CONLL_PROCESSED_TEST_DATASET),
]


ConllEntry = Tuple[List[str], List[str]]
ConllType = List[ConllEntry]


def process_json(file_name: str) -> ConllType:
    dataset = []
    with open(file_name) as json_file:
        for json_line in json_file:
            json_data = json.loads(json_line)
            sentences = json_data['sentences']
            ner = json_data['ner']

            # ner in the format start, end, tag. start and end
            # are inclusive
            total_len = 0
            for s_i, (sentence, ner_sentence) in enumerate(zip(sentences, ner)):
                curr_sentence = []
                curr_output = []
                for w_i, word in enumerate(sentence):
                    curr_sentence.append(word)
                    curr_output.append('O')
                
                assert len(curr_output) == len(sentence)
                assert len(curr_sentence) == len(sentence)
                
                for n_i, ner_entry in enumerate(ner_sentence):
                    start, end, label = ner_entry
                    start = start - total_len
                    end = end - total_len
                    curr_output[start] = "{}-{}".format('B', label)
                    inside_tag = '{}-{}'.format('I', label)
                    for c_i in range(start + 1, end + 1):
                        curr_output[c_i] = inside_tag
                    # curr_output[start + 1 : end + 1] = ['{}-{}'.format('I', label)]


                total_len += len(sentence)
                dataset.append(
                    (
                        curr_sentence,
                        curr_output,
                    )
                )
    return dataset

def serialize(conllized_input: ConllType, output_file: str) -> None:
    '''
    Given a conllized input, will convert the input into an output file that stores
    all the text in the same format as conll.
    '''
    with open(output_file, 'w') as f:
        for sentence in tqdm(conllized_input):
            words, tags = sentence
            for word, tag in zip(words, tags):
                f.write("{word}\t{tag}\n".format(
                    word=word,
                    tag=tag,
                ))
            f.write('\n')
    pass

def main() -> None:
    for _, (input_file, output_file) in enumerate(SCIERC_FILES):
        print("{} -> {}".format(input_file, output_file))
        conll_data = process_json(input_file)
        serialize(conll_data, output_file)
        print("Finished {} -> {}".format(input_file, output_file))

if __name__ == "__main__":
    main()