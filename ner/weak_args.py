import argparse
import time

from ner import active_args

def get_arg_parser() -> argparse.ArgumentParser:
    '''
    Create arg parse for active learning training containing options for
        optimizer
        hyper parameters
        model saving
        log directories
    '''
    parser = active_args.get_arg_parser()

    # Parser data loader options
    parser.add_argument('--weak_weight', type=float, default=0.01, help='The influence of the weak set')
    parser.add_argument('--weak_dictionary', type=str, default='word', help='The dictionary model to use [word, phrase]')

    return parser