from typing import (
    Dict,
    List,
    Tuple,
)

import datetime
import argparse
from xmlrpc.server import SimpleXMLRPCServer
import xmlrpc.client


from ner.trainer import Trainer
from ner import conlldataloader

def main_train(d: Dict[str, object], num_epochs: int, trainer_args, trainer_kwargs, database_items):
    print('Started main train')
    def _epoch_comparator(incoming, best) -> bool:
            res = incoming['train_f1_avg'] > best['train_f1_avg']
            if res:
                print("Found better!")
            return res
    trainer = Trainer(
        train_label_fn=lambda data, index : (data[index][0], data[index][1][0], data[index][1][1]),
        test_label_fn=lambda data, index: (data[index][0], data[index][1][0], data[index][1][1]),
        epoch_comparator=_epoch_comparator,
        verbose_log=False,
        logger=None,
        *trainer_args,
        **trainer_kwargs,
    )

    best_epoch, best_epoch_summary = trainer.train(epochs=num_epochs, update_dict=d)

    return best_epoch, best_epoch_summary, trainer.get_best_model(), ner.utils.compute_labels(
        trainer.get_best_model(),
        conlldataloader.get_unlabeled_data_loader(
            vocab=trainer_kwargs['vocab'],
            categories=trainer_kwargs['tags'],
            unlabeled_data=database_items,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            unlabeled_example_fn=lambda dataset, index: (dataset[index][0], dataset[index][1][0]),
            collate_fn=conlldataloader.collate_unlabeld_fn_with_sid,
        ),
        tag_vocab=trainer_kwargs['tags'],
        verbose=True,
        device=trainer_kwargs['device'],
    )

def get_args() -> argparse.ArgumentParser:
    '''
    Return CLI configuration for running server
    '''
    parser = argparse.ArgumentParser(description='Run the Active Learning Server to Support the application')
    parser.add_argument(
        '--cuda',
        action='store_true',
        help='Allow the Server to use the machines GPU'
    )
    parser.add_argument(
        '--load_state',
        action='store_true',
        help='Allow the Server to load the most recently saved state'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Launch debug version of server',
    )
    parser.add_argument(
        '--port',
        default=5000,
        type=int,
        help='the port to launch server on'
    )

    return parser


def setup_server():
    args = get_args().parse_args()
    server = SimpleXMLRPCServer(("localhost", args.port))
    print(f"Listening on port {args.port}...")
    server.register_function(main_train, "main_train")
    server.serve_forever()

if __name__ == "__main___":
    setup_server()
