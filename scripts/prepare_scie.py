import os
import sys
import pickle
from typing import List, Tuple, Dict
import pprint
import nltk
from tqdm import tqdm
import random

from ner import constants

# each scierc file contains a .txt for the original
# a .ann for the annotation and a .xml for the structure
SCIERCFileDefinitionType = Dict[str, str]

# maps the name of the file to all the extensions
# i.e. A00-1024 : {
#   'txt': A00-1024.txt,
#   'ann': A00-1024.ann,
#   'xml': A00-1024.txt.xml,
# }
SCIERCDirectoryDefinitionType = Dict[str, SCIERCFileDefinitionType]

# Returns train_set definition, valid_set definition
SCIERCDatasetType = Tuple[SCIERCDirectoryDefinitionType, SCIERCDirectoryDefinitionType]

# Types for conll representations of the dataset. In particular
# Each entry represents a sentence that contains the raw sentence
# and the BIO tag for the sentence
ConllEntry = Tuple[List[str], List[str]]
ConllType = List[ConllEntry]

# Stores the annotation entries as (Tag, Start Index, End Index, Word)
AnnotationType = Tuple[str, int, int, str]

def create_file_structure(all_files: List[str]) -> SCIERCDirectoryDefinitionType:
    output = {}

    for file_name in all_files:
        prefix_index = file_name.index('.')
        _, extension = os.path.splitext(file_name)
        prefix = file_name[:prefix_index]
        if prefix not in output:
            output[prefix] = {}
        
        output[prefix][extension] = file_name
    
    return output

def get_scierc_files(directory: str) -> SCIERCDirectoryDefinitionType:
    '''
    Gets a list of all the scierc files, where
    '''
    all_files = [
        file_name for file_name in os.listdir(directory)
    ]

    file_struct = create_file_structure(all_files)

    keys = list(file_struct.keys())
    random.shuffle(keys)
    split_index = int(len(keys) * 0.8)
    train_keys = keys[:split_index] # 80% in train
    valid_keys = keys[split_index:] # 20% in valid

    train_struct = {
        key: file_struct[key]
        for key in train_keys
    }

    valid_struct = {
        key: file_struct[key]
        for key in valid_keys
    }
    return train_struct, valid_struct

def parse_annotation_file(annotation_file: str) -> List[AnnotationType]:
    output = []
    with open(annotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            tokens = line.split()
            parse_tokens = tokens[:4] + [' '.join(tokens[4:])]
            relation, tag, start, end, entity = parse_tokens
            if relation[:1] != 'T':
                break

            output.append(
                (tag, int(start), int(end), entity,)
            )
    return output

def is_special(word: str) -> bool:
    return (
        len(word) > 4 and
        word[:2] == '**' and
        word[-2:] == '**'
    )

def conllize_file(raw_text_file: str, annotation_file: str) -> ConllType:
    annotation_data = parse_annotation_file(annotation_file)
    with open(raw_text_file, 'r') as content_file:
        raw_text = content_file.read()
    
    output = []
    for i in range(len(raw_text)):
        for tag, start, end, entity in annotation_data:
            if start == i:
                output.append('**{}** '.format(tag))

            if end == i:
                output.append(' **{}**'.format(tag))
        output.append(raw_text[i])

    tagged_str = ''.join(output)
    tagged_sentences = nltk.sent_tokenize(tagged_str)
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

def conllize(file_structure: SCIERCDirectoryDefinitionType, directory: str) -> ConllType:
    '''
    Given the SCIERC directory definitoin, converts the input to a conllized type
    '''
    output = []
    for file_prefix in tqdm(file_structure):
        raw_text_file = os.path.join(
            directory,
            file_structure[file_prefix]['.txt'],
        )

        annotation_file = os.path.join(
            directory,
            file_structure[file_prefix]['.ann']
        )

        output.extend(conllize_file(raw_text_file, annotation_file))
    return output

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

def main() -> None:
    pp = pprint.PrettyPrinter(indent=4)
    train_file_structure, valid_file_structure = get_scierc_files(constants.SCIERC_DIR)

    # create conll train
    conll_data = conllize(train_file_structure, constants.SCIERC_DIR)
    print("Sentences in train: {}".format(len(conll_data)))
    serialize(conll_data, constants.SCIERC_CONLL_DATASET)

    # create conll valid
    valid_conll_data = conllize(valid_file_structure, constants.SCIERC_DIR)
    print("Sentences in valid: {}".format(len(valid_conll_data)))
    serialize(valid_conll_data, constants.SCIERC_CONLL_VALID_DATASET)

if __name__ == "__main__":
    main()
