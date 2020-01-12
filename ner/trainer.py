# python imports
from typing import (
    List,
    Tuple,
    Dict,
)
import copy

# library imports
import torch
from torch import nn
from tqdm import tqdm

# local imports
try: # pragma: no cover
    import conlldataloader
    import utils
    import constants
    import vocab
    from models import dictionary_model
except: # pragma: no cover
    from . import (
        conlldataloader,
        utils,
        constants,
        vocab,
    )

    from .models import dictionary_model


'''
Epoch summary contains the following keys:
    train_loss_avg => float,
    valid_accuracy => float,
    train_accuracy => float,
    valid_f1_avg => float,
    train_f1_avg => float,
    valid_f1 => dict,
    train_f1 => dict,
'''
EpochSummaryType = Dict[str, object]


def default_epoch_comparator(incoming: dict, best: dict) -> bool:
    # return true if incoming is better than best
    return incoming['valid_f1_avg'] > best['valid_f1_avg']

class Trainer(object):

    def __init__(
        self,
        # training options
        model: nn.Module,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
        optimizer_type: str,
        vocab: vocab.Vocab,
        tags: vocab.Vocab,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        train_dataset: conlldataloader.ConllDataSet,
        test_dataset: conlldataloader.ConllDataSet,
        logger: object,

        # logging options
        train_label_fn: callable = conlldataloader.default_label_fn,
        test_label_fn: callable = conlldataloader.default_label_fn,
        epoch_comparator: callable = default_epoch_comparator,
        device: str = 'cpu',
        verbose_print: bool = True,
        verbose_log: bool = True,
        threshold: int = 100,
        train_weight_fn: callable = None,
    ):
        super(Trainer, self).__init__()

        self.model = model
        self.best_model = copy.deepcopy(model)
        if not isinstance(model, dictionary_model.DictionaryModel):
            if optimizer_type == 'SGD':
                self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
            elif optimizer_type == 'ADAM':
                self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            else:
                raise Exception("Unknown Type: {}".format(optimizer_type))
        else:
            self.optimizer = None
        
        self.verbose_print = verbose_print
        self.verbose_log = verbose_log
        self.device = device
        self.logger = logger
        self.threshold = threshold

        self.vocab = vocab
        self.tags = tags
        self.train_data_loader = conlldataloader.get_data_loader(
            vocab,
            tags,
            train_dataset,
            batch_size,
            shuffle,
            num_workers,
            label_fn=train_label_fn,
            weight_fn=train_weight_fn,
        )

        self.test_data_loader = conlldataloader.get_data_loader(
            vocab,
            tags,
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            label_fn=test_label_fn,
        )

        self.train_eval_data_loader = conlldataloader.get_data_loader(
            vocab,
            tags,
            train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            label_fn=train_label_fn,
        )

        self.epoch_comparator = epoch_comparator

        self.train_summary = []
    
    def _log_epoch_summary(self, epoch_summary: EpochSummaryType, epoch_number: int) -> None:
        utils.log_metrics(self.logger, epoch_summary['valid_f1'], "valid", epoch_number + 1)
        utils.log_metrics(self.logger, epoch_summary['train_f1'], "train", epoch_number + 1)
        self.logger.scalar_summary("train loss_avg", epoch_summary['train_loss_avg'], (epoch_number + 1))
        self.logger.scalar_summary("valid_accuracy", epoch_summary['valid_accuracy'], (epoch_number + 1))
        self.logger.scalar_summary("train_accuracy", epoch_summary['train_accuracy'], epoch_number + 1)
        self.logger.scalar_summary("valid_f1_avg", epoch_summary['valid_f1_avg'], epoch_number + 1)
        self.logger.scalar_summary("train_f1_avg", epoch_summary['train_f1_avg'], epoch_number + 1)
    
    def compute_summary(self) -> EpochSummaryType:
        '''
        Compute F1 summary for the current model
        '''
        f1_data, acc = utils.compute_f1_dataloader(
                self.model,
                self.test_data_loader,
                self.tags,
                device=self.device,
                threshold=self.threshold
            )

        f1_data_train, acc_train = utils.compute_f1_dataloader(
            self.model,
            self.train_eval_data_loader,
            self.tags,
            device=self.device,
            threshold=self.threshold,
        )

        f1_avg_train = utils.compute_avg_f1(f1_data_train)
        f1_avg_valid = utils.compute_avg_f1(f1_data)

        summary = {
            "valid_accuracy": acc,
            "train_accuracy": acc_train,
            "valid_f1_avg": f1_avg_valid,
            "train_f1_avg": f1_avg_train,
            "valid_f1": f1_data,
            "train_f1": f1_data_train,
        }

        return summary
    
    def train_epoch(self, epoch_number: int, total_epochs: int) -> EpochSummaryType:
        loss_sum = 0.0
        self.model.train()

        generator_data = self.train_data_loader
        pbar = generator_data
        if self.verbose_print:
            generator_data = tqdm(generator_data)
            pbar = generator_data.__enter__()
        for i, (s_ids, x, x_chars, y, weight) in enumerate(pbar):
            s_ids, x, x_chars, y, weight = s_ids.to(self.device), x.to(self.device), x_chars.to(self.device), y.to(self.device), weight.to(self.device)

            if isinstance(self.model, dictionary_model.DictionaryModel):
                self.model.add_example(x.long(), y.long())
                continue

            self.model.zero_grad()
            model_loss = self.model.compute_mle(x, x_chars, y, s_ids=s_ids)
            loss = torch.mean(model_loss * weight)
            loss.backward() # backpropogate
            # loss = torch.mean(model_loss)
            self.optimizer.step() # update parameters
            loss_sum += loss.item()

            if self.verbose_print:
            # update TQDM bar
                pbar.set_postfix(
                    loss_avg=loss_sum/(i + 1),
                    epoch="{}/{}".format(epoch_number + 1, total_epochs)
                )

                pbar.refresh()
            
        if self.verbose_print:
            pbar.__exit__()

        self.model.eval()
        summary = self.compute_summary()
        self.model.train()
        loss_sum /= len(self.train_data_loader)

        summary["train_loss_avg"] = loss_sum
        summary["epoch_number"] = epoch_number + 1

        # printing
        if self.verbose_print:
            utils.print_f1_summary(summary["valid_f1"], "F1 Valid Summary")
            utils.print_f1_summary(summary["train_f1"], "F1 Train Summary")
            print("Valid F1: {} | Train F1: {}".format(summary["valid_f1_avg"], summary["train_f1_avg"]))

        if self.verbose_log:
            self._log_epoch_summary(
                epoch_summary=summary, 
                epoch_number=epoch_number,
            )

        return summary
    
    def train(self, epochs: int, update_dict: Dict[str, object] = None) -> Tuple[int, EpochSummaryType]:
        best_summary = None
        best_epoch = -1
        self.train_summary = []
        self.best_model = copy.deepcopy(self.model)
        for e in range(epochs):
            epoch_summary = self.train_epoch(e, epochs)
            self.train_summary.append(epoch_summary)
            if best_summary is None or self.epoch_comparator(epoch_summary, best_summary):
                best_summary = epoch_summary
                best_epoch = e
                self.best_model = copy.deepcopy(self.model)
            if update_dict is not None:
                update_dict['train_progress'] = self.train_summary
        self.best_epoch, self.best_summary = best_epoch, best_summary
        return best_epoch, best_summary
    
    def get_progress(self) -> List[Dict[str, object]]:
        '''
        if len(self.train_summary) == num_epochs, training is finished
        '''
        return self.train_summary
    
    def get_best_model(self) -> nn.Module:
        return self.best_model
