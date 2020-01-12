import argparse
import os

try: # pragma: no cover
    # CONLL2003 Imports
    import conlldataloader
    import conll_utils

    # SCIERC Imports
    import scierc_dataloader
    import scierc_utils

    # CADEC imports
    import cadec_utils

    import constants
except: # pragma: no cover
    from . import (
        conlldataloader,
        conll_utils,
        scierc_dataloader,
        scierc_utils,
        cadec_utils,
        constants,
    )

def load_dataset(
    args: object,
    force_load: bool = False,
) -> tuple:
    module = None
    if args.dataset == 'CONLL':
        module = conll_utils
    elif args.dataset == 'SCIERC':
        module = scierc_utils
    elif args.dataset == 'CADEC':
        module = cadec_utils
    else:
        raise Exception("Unknown dataset: {}".format(
            args.dataset,
        ))

    if force_load or args.load:
        train_dataset, valid_dataset, train_vocab, output_categories = module.load()
    elif args.save:
        if not os.path.exists(constants.SAVED_DIR):
            os.makedirs(constants.SAVED_DIR)
        train_dataset, valid_dataset, train_vocab, output_categories = module.create_datasets()
        module.save(train_dataset, valid_dataset, train_vocab, output_categories)
    elif args.clean:
        module.clean_saved()
        return
    return train_dataset, valid_dataset, train_vocab, output_categories