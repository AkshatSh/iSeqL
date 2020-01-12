import os
import sys
import pickle
import argparse
from typing import List, Tuple, Dict, Callable
import json
import pprint
import nltk
import csv
from tqdm import tqdm
import random

from ner.constants import (
    CADEC_DIR,
    CADEC_CONLL_TRAIN_DATASET,
    CADEC_CONLL_VALID_DATASET,
    CADEC_CONLL_POST_TRAIN_DATASET,
    CADEC_CONLL_POST_VALID_DATASET,
    CADEC_CONLL_TRAIN_FINAL,
    CADEC_CONLL_VALID_FINAL,
    CADEC_CONLL_TEST_FINAL,
)

''' 
CADEC_DIR/text -> contains a file ending with .txt with the original text
CADEC_DIR/orignal -> conatins a file ending with .ann with the annotation

.ann file follows the format:
T{id} {category} {start_char} {end_char} {substring itself}
#{id} {annotator notes} {notes}

We only care about the first one, so we must parse the annotation file and extract the information
of only the lines that being with T{id}.

The structure is very similar to SCIERC, so we can follow similar algorithms
'''

# maps txt to the text file location, maps ann to the annotation file location
# one for each file in the dataset
CADECFileDefinitionType = Dict[str, str]

# for each of the files in the dataset maps the name of the file to the
# CADEC file description defined above
CADECDirectoryDefinitionType = Dict[str, CADECFileDefinitionType]

# contains information about the train and test split
CADECDatasetType = Tuple[CADECDirectoryDefinitionType, CADECDirectoryDefinitionType]

# (category, start index, end index, actual annotation)
AnnotationType = Tuple[str, int, int, str]

# Types for conll representations of the dataset. In particular
# Each entry represents a sentence that contains the raw sentence
# and the BIO tag for the sentence
ConllEntry = Tuple[List[str], List[str]]
ConllType = List[ConllEntry]

def is_special(word: str) -> bool:
    '''
    Check if `word` is one of the special words in our
    custom parse
    '''
    return (
        len(word) > 4 and
        word[:2] == '**' and
        word[-2:] == '**'
    )

def is_number(s: str) -> bool:
    '''
    check if the passed in `s` is a number
    '''
    try:
        int(s)
        return True
    except ValueError:
        return False

def parse_with_extension(directory_location: str, directory_definition: CADECDirectoryDefinitionType) -> None:
    '''
    populates the directory definition argument with information about all the files with their extension
    '''
    all_files = [file_name for file_name in os.listdir(directory_location)]
    for file_name in all_files:
        raw_name, extension = os.path.splitext(file_name)
        if raw_name not in directory_definition:
            directory_definition[raw_name] = {}
        directory_definition[raw_name][extension] = os.path.join(directory_location, file_name)

def parse_directiory(directory_location: str) -> CADECDirectoryDefinitionType:
    '''
    parses the directory into a format meant for processing
    '''
    output = {}
    parse_with_extension(os.path.join(directory_location, 'text'), output)
    parse_with_extension(os.path.join(directory_location, 'original'), output)
    return output

def create_splits(directory_def: CADECDirectoryDefinitionType, ratio: float) -> CADECDatasetType:
    '''
    Splits the dataset and returns a definition of the newly created dataset with train and test
    splits
    '''
    keys = list(directory_def.keys())
    random.shuffle(keys)
    split_index = int(ratio * len(keys))
    train_keys = keys[:split_index]
    valid_keys = keys[split_index:]
    return (
        {key: directory_def[key] for key in train_keys},
        {key: directory_def[key] for key in valid_keys},
    )

def merge_intervals(intervals: List[object], get_interval: callable, set_interval: callable) -> List[Tuple[int, int]]:
    sorted_by_lower_bound = sorted(intervals, key=lambda tup: get_interval(tup)[0])
    merged = []

    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            # test for intersection between lower and higher:
            # we know via sorting that lower[0] <= higher[0]
            if get_interval(higher)[0] <= get_interval(lower)[1]:
                upper_bound = max(get_interval(lower)[1], get_interval(higher)[1])
                merged[-1] = set_interval(merged[-1], get_interval(lower)[0], upper_bound)  # replace by merged interval
            else:
                merged.append(higher)
    return merged

def parse_annotation_file(annotation_file: str) -> List[AnnotationType]:
    output = []
    with open(annotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith("T"):
                # only care about the actual annotations
                continue
            tokens = line.split()
            '''
            Example
            T1	ADR 0 6;13 18	Muscle pains
            T2	ADR 7 12;13 18	joint pains
            T3	ADR 20 34	gastric upsets
            '''
            # ignore term id
            tokens = tokens[1:]
            category = tokens[0]
            ranges = []
            start = None
            for i in range(1, len(tokens)):
                curr = tokens[i]
                if is_number(curr):
                    if start is None:
                        start = curr
                    elif start is not None:
                        ranges.append((int(start), int(curr)))
                        start = None
                elif ';' in curr:
                    potential_split = curr.split(';')
                    if len(potential_split) == 2 and all([is_number(token) for token in potential_split]):
                        ranges.append((int(start), int(potential_split[0])))
                        start = potential_split[1]
                    else:
                        break
                else:
                    break
            label = " ".join(tokens[i:])
            for index_range in ranges:
                output.append((category, index_range[0], index_range[1], label))
    def _set_interval(obj: object, start: int, end: int):
        return (
            obj[0],
            start,
            end,
            obj[3],
        )
    output = merge_intervals(
        output,
        get_interval=lambda obj: (obj[1], obj[2]),
        set_interval=_set_interval,
    )

    return output

def conllize_file(raw_text_file: str, annotation_file: str, post_level: bool = False) -> ConllType:
    annotation_data = parse_annotation_file(annotation_file)
    with open(raw_text_file, 'r') as content_file:
        raw_text = content_file.read()
    
    output = []
    for i in range(len(raw_text)):
        for tag, start, end, entity in annotation_data:
            if start == i:
                output.append(' **{}** '.format(tag))

            if end == i:
                output.append(' **{}** '.format(tag))
        output.append(raw_text[i])

    tagged_str = ''.join(output)
    if len(tagged_str) > 0:
        tagged_sentences = nltk.sent_tokenize(tagged_str) if not post_level else [tagged_str]
    else:
        tagged_sentences = []
    outputs = []
    for tagged_sentence in tagged_sentences:
        tagged_word_tokens = nltk.word_tokenize(tagged_sentence)
        
        tagged = []
        raw = []

        last_tag = None
        is_start = False
        for i, word in enumerate(tagged_word_tokens):
            if is_special(word):
                if last_tag is None:
                    # start of tagged sequence
                    tag = word[2:-2]
                    last_tag = tag
                    is_start = True
                else: 
                    # end of tagged sequence
                    last_tag = None
            else:
                raw.append(word)
                if last_tag is None:
                    # not in tagged sequence
                    tagged.append('O')
                else:
                    # in tagged sequence
                    tag_prefix = 'B' if is_start else 'I'
                    is_start = False
                    tagged.append("{prefix}-{tag}".format(prefix=tag_prefix, tag=last_tag))
        assert len(raw) == len(tagged)
        outputs.append((raw, tagged))
    return outputs

def conllize(file_structure: CADECDirectoryDefinitionType, post_level: bool = False) -> ConllType:
    outputs = []
    for raw_file_name in tqdm(file_structure):
        annotation_file = file_structure[raw_file_name]['.ann']
        raw_text_file = file_structure[raw_file_name]['.txt']
        conll_out = conllize_file(raw_text_file, annotation_file, post_level=post_level)
        outputs.extend(conll_out)
    return outputs

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

def create_csv_data(raw_file_name: str, post_level: bool = False, csv_data: List[Tuple[int, str]] = None) -> List[Tuple[int, str]]:
    with open(raw_file_name, 'r') as content_file:
        raw_text = content_file.read()
    if csv_data == None:
        csv_data = []
    entry_id = csv_data[-1][0] + 1 if len(csv_data) > 0 else 1000
    proc_text = nltk.sent_tokenize(raw_text) if not post_level else [raw_text]
    for entry in proc_text:
        csv_data.append((entry_id, entry))
        entry_id += 1
    return csv_data

def create_label_csv_data(raw_text_file: str, annotation_file: str, post_level: bool, csv_data: list = [],):
    label_csv_data = conllize_file(
        raw_text_file=raw_text_file,
        annotation_file=annotation_file,
        post_level=post_level,
    )

    entry_id = csv_data[-1][0] + 1 if len(csv_data) > 0 else 1000
    for conll_data in label_csv_data:
        raw_sentence: List[str] = None
        labeled_sentence: List[str] = None
        raw_sentence, labeled_sentence = conll_data
        raw_sentence_str = ' '.join(raw_sentence)
        labeled_sentence_str = ' '.join(labeled_sentence)
        csv_data.append((entry_id, raw_sentence_str, labeled_sentence_str))
        entry_id += 1
    
    return csv_data


def serialize_csv_data(csv_file_name, file_structure: CADECDirectoryDefinitionType, post_level: bool = False, add_labels: bool = False):
    csv_data = []
    for raw_file_name in tqdm(file_structure):
        annotation_file = file_structure[raw_file_name]['.ann']
        raw_text_file = file_structure[raw_file_name]['.txt']
        if not add_labels:
            csv_data = create_csv_data(raw_text_file, post_level=post_level, csv_data=csv_data)
        else:
            csv_data = create_label_csv_data(
                raw_text_file=raw_text_file,
                annotation_file=annotation_file,
                post_level=post_level,
                csv_data=csv_data
            )
    with open(csv_file_name, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for csv_entry in csv_data:
            writer.writerow(csv_entry)
    return

def main():
    parser = argparse.ArgumentParser(description='Generate CADEC dataset files')
    parser.add_argument('--post_level', action='store_true', help='generate post level dataset')
    parser.add_argument('--raw_csv', action='store_true', help='process the dataset to a raw csv')
    parser.add_argument('--out_csv', type=str, help='name of the output csv file')
    parser.add_argument('--add_labels', action='store_true', help='will produce a 3rd column storing the labels')
    args = parser.parse_args()
    random.seed(1234)
    cadec_dir_def = parse_directiory(CADEC_DIR)
    if args.raw_csv:
        serialize_csv_data(args.out_csv, cadec_dir_def, args.post_level, args.add_labels)
        return
    cadec_dataset = create_splits(cadec_dir_def, ratio=0.8)
    train_valid_split = create_splits(cadec_dataset[0], ratio=0.8)
    # annotation_info = parse_annotation_file("/Users/akshatshrivastava/Documents/UniversityofWashington/Research/ActiveLearnedNER/data/cadec/original/LIPITOR.977.ann")
    # print(annotation_info)
    # outputs = conllize_file(
    #     raw_text_file="/Users/akshatshrivastava/Documents/UniversityofWashington/Research/ActiveLearnedNER/data/cadec/text/LIPITOR.977.txt",
    #     annotation_file="/Users/akshatshrivastava/Documents/UniversityofWashington/Research/ActiveLearnedNER/data/cadec/original/LIPITOR.977.ann",
    # )
    # for out in outputs: 
    #     print('=' * 30)
    #     for word, tag in zip(out[0], out[1]):
    #         print("{} - {}".format(word, tag))
    serialize(
        conllized_input=conllize(cadec_dataset[0], post_level=args.post_level),
        output_file=CADEC_CONLL_TRAIN_DATASET if not args.post_level else CADEC_CONLL_POST_TRAIN_DATASET,
    )

    serialize(
        conllized_input=conllize(cadec_dataset[1], post_level=args.post_level),
        output_file=CADEC_CONLL_VALID_DATASET if not args.post_level else CADEC_CONLL_POST_VALID_DATASET,
    )


if __name__ == "__main__":
    main()

