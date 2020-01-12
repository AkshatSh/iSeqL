from torch.multiprocessing import Process, Manager, set_start_method, Queue, Pool
import torch.multiprocessing as mp
# try:
#     mp.set_start_method('forkserver', force=True)
#     print('set forkserver')
# except RuntimeError as e:
#     print(e)
import threading
import os
import sys

from typing import (
    List,
    Tuple,
    Dict,
    Any,
)
import copy
import torch
import ner
from ner.trainer import Trainer
from ner import conlldataloader


class ProgressableThread(threading.Thread):
    def get_progress(self) -> Dict[str, object]:
        '''
        return a dictionary representing the progress of
        the current thread
        '''
        return {}
    
    def get_thread_id(self) -> int:
        return threading.get_ident()
    
    def run(self):
        raise NotImplementedError(
            "Implement thi method"
        )

def main_q(pqueue, out_pqueue, d):
    print('Started main train q')
    total = 0
    res = [None, None, None, None, None]
    while total < len(res):
        item = pqueue.get()
        res[total] = item
        total+=1
    dtemp, num_epochs, trainer_args, trainer_kwargs, database_items = res

    # model = trainer_kwargs['model']
    # model.share_memory()
    # trainer_kwargs['model'] = model

    best_epoch, best_epoch_summary, model, labels = main_train(d, num_epochs, trainer_args, trainer_kwargs, database_items)

    print('starting outputting results')
    out_pqueue.put(best_epoch)
    out_pqueue.put(best_epoch_summary)
    out_pqueue.put(model.state_dict())
    out_pqueue.put(labels)
    # d['best_epoch'] = best_epoch
    # d['best_epoch_summary'] = best_epoch_summary
    # d['model'] = model
    # d['labels'] = labels

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

def get_queue_dict(queue: mp.Queue, item_names: List[str]) -> Dict[str, Any]:
    res = [None] * len(item_names)
    for i in range(res):
        res[i] = queue.get()

    return {
        name: value
        for (name, value) in zip(item_names, res)
    }

class TrainingThread(ProgressableThread):
    def set_trainer(self, trainer: Trainer):
        self.trainer = trainer
    
    def set_fork_pool(self, fork_pool: Pool):
        self.fork_pool = fork_pool
    
    def set_trainer_args(self, *args, **kwargs):
        self.trainer_args = args
        self.trainer_kwargs = kwargs
    
    def set_host(self, host: object):
        self.d = {'train_progress': []}
        self.host = host
    
    def set_complete_func(self, func: callable):
        self.complete_func = func
    
    def set_num_epochs(self, num_epochs):
        self.num_epochs = num_epochs
    
    def get_progress(self) -> Dict[str, object]:
        return {
            "train_progress": self.d['train_progress'],
            "num_epochs": self.num_epochs
        }
    
    def set_database_items(self, dbi):
        self.database_items = dbi
    
    def run(self):
        # mp.set_start_method('spawn', force=True)
        with Manager() as manager:
            d = manager.dict()
            self.d = d
            d['train_progress'] = []
            d['best_epoch'] = None
            d['best_epoch_summary'] = None
            d['model'] = None
            d['labels'] = None
            pqueue = mp.Queue()
            out_pqueue = mp.Queue()
            model = self.trainer_kwargs['model']
            # model = copy.deepcopy(model)
            model.share_memory()
            self.trainer_kwargs['model'] = model
            self.trainer_kwargs['num_workers'] = 0
            p = Process(target=main_q, args=(pqueue, out_pqueue, d))
            p.daemon = True
            p.start()
            # pool.apply_async(main_q, args=(pqueue, out_pqueue, d, ))
            # pool.apply_async(main_train, args=(d, self.num_epochs, self.trainer_args, self.trainer_kwargs, self.datbaase_items))
            # pool.starmap(main_q, [(pqueue, out_pqueue, d),])
            pqueue.put(None)
            pqueue.put(self.num_epochs)
            pqueue.put(self.trainer_args)
            pqueue.put(self.trainer_kwargs)
            pqueue.put(self.database_items)
            p.join()
            # pool.close()
            # pool.join()
            print('Process results: ', len(d.keys()))
            # best_epoch = d['best_epoch']
            # best_epoch_sumamry = d['best_epoch_summary']
            # model = d['model']
            # labels = d['labels']
            self.d = get_queue_dict(
                out_pqueue,
                item_names=[
                    'best_epoch',
                    'best_epoch_summary',
                    'model',
                    'labels',
                ]
            )
            best_epoch = self.d['best_epoch']
            best_epoch_sumamry = self.d['best_epoch_summary']
            model = model.load_state_dict(self.d['model'])
            labels = self.d['labels']

            self.d = {
                "train_progress": d['train_progress'],
            }
        # best_epoch, best_epoch_summary = self.trainer.train(epochs=self.num_epochs)
        self.complete_func(self.host, {
            "best_epoch": best_epoch,
            "best_epoch_summary": best_epoch_sumamry,
            "model": model,
            "labels": labels,
        })
