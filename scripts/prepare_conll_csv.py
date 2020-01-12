from tqdm import tqdm
import os
import csv
import argparse
from typing import (
    List,
)

from ner import (
    conll_utils,
    constants,
    conlldataloader
)

def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, required=True, help='the location csv file to output')
    parser.add_argument('--sentence_level', action='store_true', help='should use sentence level dataset otherwise document level')
    return parser

def document_level_writer(
    writer: csv.writer,
    train_dataset: conlldataloader.ConllDataSet,
    start_id: int = 0, 
) -> None:

    def is_document_start(input_word_arr: List[str]) -> bool:
        return len(input_word_arr) == 1 and input_word_arr[0] == '-DOCSTART-'

    curr_document = []
    document_id = start_id
    for i, data_point in enumerate(tqdm(train_dataset)):
        input_arr = data_point['input']
        input_word_arr = [point[constants.CONLL2003_WORD] for point in input_arr]
        if is_document_start(input_word_arr) and len(curr_document) > 0:
            writer.writerow([document_id, ' '.join(curr_document)])
            document_id += 1
            curr_document = []
        else:
            input_str = ' '.join(input_word_arr)
            curr_document.append(input_str)

def sentence_level_writer(
    writer: csv.writer,
    train_dataset: conlldataloader.ConllDataSet,
    start_id: int = 0,
) -> None:
    for i, data_point in enumerate(tqdm(train_dataset)):
            input_arr = data_point['input']
            input_word_arr = [point[constants.CONLL2003_WORD] for point in input_arr]
            input_str = ' '.join(input_word_arr)
            input_id = start_id + i
            writer.writerow([input_id, input_str])

def main():
    args = get_arg_parser().parse_args()
    train_dataset, valid_dataset, train_vocab, output_vocab = conll_utils.load()
    start_id = 2000
    with open(args.output_file, 'w') as f:
        writer = csv.writer(f)
        if args.sentence_level:
            sentence_level_writer(
                writer,
                train_dataset,
                start_id,
            )
        else:
            document_level_writer(
                writer,
                train_dataset,
                start_id,
            )


if __name__ == "__main__":
    main()