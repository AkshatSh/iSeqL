from typing import (
    List,
    Tuple,
    Dict,
    Callable,
)
import os
import sys
import copy
import torch
import time
from torch.multiprocessing import Pool
import pickle

sys.path.append("..")

import ner
from ner import (
    trainer,
    active_heuristic,
    vocab,
    conlldataloader
)

from ner.tensor_logger import Logger

from ner.constants import (
    ACTIVE_LEARNING_SAMPLE,
    ACTIVE_LEARNING_TOP_K,
)

from ner.models.cached_bilstm_crf import (
    CachedBiLSTMCRF,
)

from ner.models.cached_embedder import (
    CachedEmbedder,
)

from ner.models.elmo import (
    FrozenELMo,
)

# from . import database_manager
# from . import constants

import database_manager
import constants
import utils
import progress_thread
import thread_manager

from configurations.configuration import Configuration

HEURISTIC_CLASSES = [
    active_heuristic.Random,
    active_heuristic.KNNEmbeddings,
    active_heuristic.Uncertantiy,
]

HEURISTIC_NAMES = {
    "RANDOM": 0,
    "KNN": 1,
    "UNCERTAINTY": 2,
}

# DEFAULT_AL_CONFIG = Configuration('active_learning_manager_configuration.json')
DEFAULT_AL_CONFIG = Configuration('yelp_al_config.json')

'''
The core class powering the server, this handles all the active learning pipeline

Keeps track of:
    * Model: The model that will be trained on this dataset
    * Dataloaders: valid and train dataloaders for building a model
    * Vocabulary Objects: the vocabulary and the output tags
    * unlabeled_corpus: 

API:
    * get_query(self) -> Tuple[int, str]
    * answer_query(self, sentence_id: int, label: List[str])
    * train(self)
    * evaluate(self)
    * update_example(self, sentence_id: int, label: List[str] )
    * move_to_test(self, sentence_id: int)
'''
class ActiveLearningManager(object):
    def __init__(
        self, 
        db: database_manager.DatabaseManager,
        thread_manager: thread_manager.ThreadManager,
        # fork_pool: Pool,
        ner_class: str,
        session_dir: str = None,
        device: str = 'cpu',
        config: Configuration = DEFAULT_AL_CONFIG,
    ):
        self.is_start = True
        self.configuration = config
        self.database = db
        self.thread_manager = thread_manager
        self.device = device
        self.prev_labels = None
        self.train_data = {}
        self.test_data = {}
        self.exclude_data = {}
        self.ner_class = ner_class
        self.tag_vocab = vocab.build_output_vocab([f'B-{ner_class}', f'I-{ner_class}', 'O'])
        self.vocab = db.vocab
        self.is_training = False
        self.fork_pool = None

        self._init_experiment_stats()
        self._init_predictions()

        self.logger = Logger(
            os.path.join(
                self.database.session_dir, 
                f'{ner_class}/'
            ),
        )

        self.model = ner.utils.build_model(
            model_type=self.configuration.get_key('model_schema/model_type'),
            embedding_dim=self.configuration.get_key('model_schema/embedding_dim'),
            hidden_dim=self.configuration.get_key('model_schema/hidden_dim'),
            vocab=self.vocab,
            tag_vocab=self.tag_vocab,
            batch_size=1,
        )

        self.samples = self.configuration.get_key('active_learning_sampling_rate')
        self.curr_sample_size = 0
        self.curr_query = None
        self.curr_query_predictions = None
        self.curr_test_query_ids = None
        self.sampling_strategy = self.configuration.get_key('sampling_strategy')
        heuristic_type = self.configuration.get_key('active_heuristic')

        heuristics = [
            h_class(vocab=self.vocab, tag_vocab=self.tag_vocab) for h_class in HEURISTIC_CLASSES
        ]

        if (heuristic_type == ner.constants.ACTIVE_LEARNING_RANDOM_H):
            self.heuristic_index = 0
        elif (heuristic_type == ner.constants.ACTIVE_LEARNING_KNN):
            self.heuristic_index = 1
        else:
            self.heuristic_index = 2

        load_res = session_dir and self.load(session_dir)
        self.total_corpus = conlldataloader.ConllDataSetUnlabeled(
            dataset=self.database.database,
            data_constructor=lambda data: [
                (s_id, sent) for s_id,(sent, label) in data.items()
            ]
        )
        if not load_res:
            self.unlabeled_corpus = conlldataloader.ConllDataSetUnlabeled(
                dataset=self.database.database,
                data_constructor=lambda data: [
                    (s_id, sent) for s_id,(sent, label) in data.items() if label is None
                ]
            )

            # self.unlabeled_corpus = conlldataloader.ConllTorchDatasetUnlabeld(
            #     vocab=self.vocab,
            #     categories=self.tag_vocab,
            #     conlldataset=list({
            #         sentence_id: sentence
            #         for sentence_id, (sentence, label) in self.database.database.items() if label is None
            #     }.items()),
            # )
            self.save(session_dir)
        else:
            print("loaded unlabeled corpus state")

        for h in heuristics:
            load_res = session_dir and h.load(session_dir)
            if not load_res:
                print("preparing active heuristic: {}".format(h))
                h.prepare(
                    model=self.model,
                    dataset=self.unlabeled_corpus, # .conll,
                    device=self.device,
                )
                h.save(session_dir)
            else:
                print("loaded active heuristic: {}".format(h))
        
        self.heuristics = heuristics
        cached_embedder = CachedEmbedder(
            embedder=FrozenELMo.instance(),
            embedding_dimensions=FrozenELMo.DIMENSIONS,
        ).to(self.device)
        self.load_cached_embeder(cached_embedder, session_dir)
        self.model = CachedBiLSTMCRF(
            vocab=self.vocab,
            tag_set=self.tag_vocab,
            hidden_dim=self.configuration.get_key('model_schema/hidden_dim'),
            batch_size=1,
            embedder=cached_embedder,
        ).to(self.device)
        print("loaded cached embedding vectors")
    
    def _init_experiment_stats(self):
        self.experiment_stats = {
            # store the time each batch completeted for analysis
            "batch_time": [],
            "batch_start_time": [],
            "batch_end_time": [],
            "used_cheatsheet": [],
            # is the experiment over
            "finished": False,
        }
    
    def _init_predictions(self):
        self.predictions = {
            # all the data about the predictions the model made
            "predicted_data": {
                s_id : (sent, 
                    {   
                        "labeled_set_sizes": [],
                        "ranges": [],
                        "entities": [],
                        "real_ranges": [],
                        "real_entities": [],
                        "is_test": False,
                        "is_train": False,
                    },
                ) for s_id, (sent, _) in self.database.database.items()
            },
            # a list of all the iterations that have occured so far
            # and the corresponding labeled set sizes
            "labeled_set_sizes": [],

            # same for training explicitly
            "training_set_sizes": [],

            # same for testing explicitly
            "testing_set_sizes": [],

            # flipped data, a list of tuples where the first
            # item is the number positive flipped and the second number
            # is the number negative flipped (see utils.py) for 
            # definition of metrics
            "flipped_data" : [],

            # training summary, a list of training summaries for each of the iterations
            "training_summary": [],

            # a reference to the entire dataset
            "entire_data": self.database.entire_database,

            # connecting sentence id to entry
            "entry_to_sentences": self.database.entry_to_sentences,

            # get sentence level data
            "sentence_data": self.database.sentence_data,
        }
    
    def load_cached_embeder(self, cached_embedder: CachedEmbedder, session_dir: str) -> bool:
        path = os.path.join(session_dir, "cached_embedder.pkl")
        if os.path.exists(path):
            print("loading cached embedding vectors")
            with open(os.path.join(path), 'rb') as f:
                save_state = pickle.load(f)
                cached_embedder.load(save_state, 'cached_embedder')
                cached_embedder.to(self.device)
        else:
            print("caching embedding vectors")
            save_state = {}
            cached_embedder.cache_dataset(self.total_corpus, verbose=True, device=self.device)
            cached_embedder.save(save_state, "cached_embedder")
            with open(os.path.join(session_dir, "cached_embedder.pkl"), 'wb') as f:
                pickle.dump(save_state, f)

    def save(self, session_dir: str):
        with open(os.path.join(session_dir, "unlabeled_corpus.pkl"), 'wb') as f:
            pickle.dump(self.unlabeled_corpus, f)
    
    def load(self, session_dir: str) -> bool:
        if not os.path.exists(os.path.join(session_dir, "unlabeled_corpus.pkl")):
            return False

        with open(os.path.join(session_dir, "unlabeled_corpus.pkl"), 'rb') as f:
            self.unlabeled_corpus = pickle.load(f)
    
    def evaluate_ground_truth(self) -> ner.trainer.EpochSummaryType:
        trainer = ner.trainer.Trainer(
            model=self.model,
            learning_rate=self.configuration.get_key('trainer_params/learning_rate'),
            weight_decay=self.configuration.get_key('trainer_params/weight_decay'),
            momentum=self.configuration.get_key('trainer_params/momentum'),
            optimizer_type=self.configuration.get_key('trainer_params/optimizer_type'),
            vocab=self.vocab,
            tags=self.tag_vocab,
            batch_size=self.configuration.get_key('trainer_params/batch_size'),
            shuffle=True,
            num_workers=self.configuration.get_key('trainer_params/num_workers'),
            train_dataset=[(s_id, self.database.database[s_id]) for s_id in self.train_data],
            test_dataset=[
                (
                    s_id,
                    self.database.ground_truth_database[s_id],
                ) for s_id in self.database.ground_truth_database
            ],
            logger=self.logger,
            device=self.device,
            verbose_print=True,
            verbose_log=True,
            train_label_fn=lambda data, index : (data[index][0], data[index][1][0], data[index][1][1]),
            test_label_fn=lambda data, index: (data[index][0], data[index][1][0], data[index][1][1]),
            epoch_comparator=None,
        )

        summary = trainer.compute_summary()
        return summary
    
    def _test_set_query_size(self, train_sample_size: int) -> int:
        '''
        Adjust the sample size to include the test_set examples
        '''
        test_set_split = self.configuration.get_key('test_set_split') # i.e 0.2
        curr_split = 1 - test_set_split # i.e 0.8
        new_samples_to_add = test_set_split / curr_split
        return max(int(train_sample_size * new_samples_to_add), 1) # at least add 1 example

    
    def get_query(self, include_labels=True) -> Tuple[Tuple[int, str], List[object]]:
        if self.curr_query is not None:
            # current query has not been labeled, wait for it to be labeled
            return self.curr_query, self.curr_query_predictions, self.curr_sample_size >= len(self.samples)
        
        if self.curr_sample_size >= len(self.samples):
            # TODO(AkshatSh): Only for Turk Study
            # finished experiment
            if not self.experiment_stats["finished"]:
                self.experiment_stats["batch_time"].append(time.time())
                self.experiment_stats["finished"] = True
            return None, None, True

        # calculate the current train set and test set query sizes
        train_sample_size = self.samples[self.curr_sample_size]
        test_sample_size = self._test_set_query_size(train_sample_size)
        if self.curr_sample_size == len(self.samples) -1:
            # last batch
            # all goes to test
            test_sample_size += train_sample_size
            train_sample_size = 0
        sample_size = train_sample_size + test_sample_size

        distribution = self.heuristics[
            self.heuristic_index
        ].evaluate(self.model, self.unlabeled_corpus, device=self.device)

        if self.sampling_strategy == ACTIVE_LEARNING_SAMPLE:
            new_points = torch.multinomial(distribution, sample_size)
        elif self.sampling_strategy == ACTIVE_LEARNING_TOP_K:
            new_points = sorted(
                range(len(distribution)),
                reverse=True,
                key=lambda ind: distribution[ind]
            )
        new_points = new_points[:sample_size]
        self.curr_sample_size += 1
        self.curr_query = [self.unlabeled_corpus[new_point] for new_point in new_points]
        self.curr_query_predictions = {
            s_id: self.predictions["predicted_data"][s_id] for (s_id, _) in self.curr_query
        }

        # default everything in train set, but last (test_sample_size) examples go to test set
        self.curr_test_query_ids = [s_id for (s_id, sent) in self.curr_query[-test_sample_size:]]
        self.is_start = False

        # new batch, so for turk include timing
        self.experiment_stats['batch_time'].append(time.time())

        self.experiment_stats["batch_start_time"].append(time.time())
    
        return self.curr_query, self.curr_query_predictions, self.curr_sample_size >= len(self.samples) # is last batch
    
    def get_is_start(self) -> bool:
        return self.is_start
    
    def train(self):

        comparison_metric = self.configuration.get_key('comparison_metric')
        def _epoch_comparator(incoming, best) -> bool:
            res = incoming[comparison_metric] > best[comparison_metric]
            if res:
                print("Found better!")
            return res
        self.trainer = ner.trainer.Trainer(
            model=copy.deepcopy(self.model),
            learning_rate=self.configuration.get_key('trainer_params/learning_rate'),
            weight_decay=self.configuration.get_key('trainer_params/weight_decay'),
            momentum=self.configuration.get_key('trainer_params/momentum'),
            optimizer_type=self.configuration.get_key('trainer_params/optimizer_type'),
            vocab=self.vocab,
            tags=self.tag_vocab,
            batch_size=self.configuration.get_key('trainer_params/batch_size'),
            shuffle=True,
            num_workers=self.configuration.get_key('trainer_params/num_workers'),
            train_dataset=[(s_id, self.database.database[s_id]) for s_id in self.train_data],
            test_dataset=[(s_id, self.database.database[s_id]) for s_id in self.test_data],
            logger=self.logger,
            device=self.device,
            verbose_print=True,
            verbose_log=True,
            train_label_fn=lambda data, index : (data[index][0], data[index][1][0], data[index][1][1]),
            test_label_fn=lambda data, index: (data[index][0], data[index][1][0], data[index][1][1]),
            epoch_comparator=_epoch_comparator,
        )

        thread = progress_thread.TrainingThread(
            name="TrainingThread_class-{}".format(self.ner_class)
        )
        thread.set_trainer(self.trainer)
        thread.set_num_epochs(self.configuration.get_key('trainer_params/num_epochs'))
        thread.set_host(self)
        thread.set_trainer_args(
            model=self.model,
            learning_rate=self.configuration.get_key('trainer_params/learning_rate'),
            weight_decay=self.configuration.get_key('trainer_params/weight_decay'),
            momentum=self.configuration.get_key('trainer_params/momentum'),
            optimizer_type=self.configuration.get_key('trainer_params/optimizer_type'),
            vocab=self.vocab,
            tags=self.tag_vocab,
            batch_size=self.configuration.get_key('trainer_params/batch_size'),
            shuffle=True,
            num_workers=self.configuration.get_key('trainer_params/num_workers'),
            train_dataset=[(s_id, self.database.database[s_id]) for s_id in self.train_data],
            test_dataset=[(s_id, self.database.database[s_id]) for s_id in self.test_data],
            # logger=self.logger,
            device=self.device,
            verbose_print=True,
            # train_label_fn=lambda data, index : (data[index][0], data[index][1][0], data[index][1][1]),
            # test_label_fn=lambda data, index: (data[index][0], data[index][1][0], data[index][1][1]),
            # epoch_comparator=_epoch_comparator,
        )

        thread.set_database_items(list(self.database.database.items()))
        # thread.set_fork_pool(self.fork_pool)

        def _complete_func(host, state):
            host.is_training = False
            host.model = state["model"]
            host.predictions["training_summary"].append(state["best_epoch_summary"])
            host.evaluate(state["labels"])
            return True
        thread.set_complete_func(_complete_func)

        thread_id = self.thread_manager.add_thread(thread)
        self.is_training = True
        # best_epoch, best_summary = self.trainer.train(epochs=5)
        # self.model = self.trainer.get_best_model()
        return thread_id
    
    def evaluate(self, labels=None):
        database_dataset = list(self.database.database.items())
        if labels is None:
            labels = ner.utils.compute_labels(
                self.model,
                conlldataloader.get_unlabeled_data_loader(
                    vocab=self.vocab,
                    categories=self.tag_vocab,
                    unlabeled_data=database_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    unlabeled_example_fn=lambda dataset, index: (dataset[index][0], dataset[index][1][0]),
                    collate_fn=conlldataloader.collate_unlabeld_fn_with_sid,
                ),
                tag_vocab=self.tag_vocab,
                verbose=True,
                device=self.device,
            )

        pos_flipped, neg_flipped = utils.compute_total_flip(self.prev_labels, labels)

        number_labeled = len(self.train_data) + len(self.test_data)

        flipped_data = {
            "labeled_set_sizes": number_labeled,
            "pos_flipped": pos_flipped,
            "neg_flipped": neg_flipped,
            "total_flipped": pos_flipped + neg_flipped,
        }

        self.prev_labels = labels

        if (
            len(self.predictions["labeled_set_sizes"]) == 0 or 
            number_labeled != self.predictions["labeled_set_sizes"][-1]
        ):
            self.predictions["flipped_data"].append(flipped_data)
            self.predictions["labeled_set_sizes"].append(number_labeled)
            self.predictions["training_set_sizes"].append(len(self.train_data))
            self.predictions["testing_set_sizes"].append(len(self.train_data))
        else:
            self.predictions["flipped_data"][-1] = (flipped_data)
            self.predictions["labeled_set_sizes"] = number_labeled
            self.predictions["training_set_sizes"] = len(self.train_data)
            self.predictions["testing_set_sizes"] = len(self.train_data)

        for i, label in enumerate(labels):
            s_id, (sent, real_label) = database_dataset[i]
            ranges, _, entities = self.explain_labels(sent, label)

            stored_sent, stored_label_info = self.predictions["predicted_data"][s_id]
            real_ranges, _, real_entities = self.explain_labels(sent, real_label)
            if stored_label_info is None or len(self.predictions["labeled_set_sizes"]) == 0:
                self.predictions["predicted_data"][s_id] = (
                    sent,
                    {   "labeled_set_sizes": [number_labeled], # len(self.train_data)],
                        "ranges": [ranges],
                        "entities": [entities],
                        "real_ranges": real_ranges,
                        "real_entities": real_entities,
                        "is_test": (s_id in self.test_data and real_label is not None),
                        "is_train": (s_id in self.train_data and real_label is not None),
                    }
                )
            else:
                if (
                    len(stored_label_info['labeled_set_sizes']) == 0 or 
                    number_labeled != stored_label_info['labeled_set_sizes'][-1]
                ):
                    stored_label_info['labeled_set_sizes'].append(number_labeled)
                    stored_label_info['ranges'].append(ranges)
                    stored_label_info['entities'].append(entities)
                else:
                    stored_label_info['labeled_set_sizes'][-1] = number_labeled #len(self.train_data)
                    stored_label_info['ranges'][-1] = ranges
                    stored_label_info['entities'][-1] = entities
                stored_label_info['is_test'] = s_id in self.test_data and real_label is not None
                stored_label_info['is_train'] = s_id in self.train_data and real_label is not None
                stored_label_info["real_ranges"] = real_ranges
                stored_label_info["real_entities"] = real_entities

        return self.predictions
    
    def get_training_progress(self):
        if hasattr(self, 'trainer'):
            return self.trainer.get_progress()
        else:
            return []
    
    def get_predictions(self):
        if len(self.predictions["training_summary"]) == 0:
            return None
        print(self.predictions.keys())
        return self.predictions

    def force_eval(self):
        f1_data = ner.utils.compute_f1_dataloader(
            self.model,
            conlldataloader.get_data_loader(
                self.vocab,
                self.tag_vocab,
                self.test_data,
                1,
                False,
                0,
                label_fn=lambda data, index: (data[index][0], data[index][1]),
            ),
            tag_vocab=self.tag_vocab,
        )

        return f1_data
    
    def update_example(self, sentence_id: int, label: List[str] ):
        sent, old_label = self.database.database[sentence_id]
        if old_label is not None and old_label == label:
            # already processed
            return
        self.database.database[sentence_id] = (sent, label)
        self.curr_query.remove((sentence_id, sent))
        self.unlabeled_corpus.remove((sentence_id, sent))
        # del self.unlabeled_corpus[sentence_id]
        if label is None:
            self.exclude_data[sentence_id] = True
            self.predictions["predicted_data"][sentence_id][1]["is_test"] = False
            self.predictions["predicted_data"][sentence_id][1]["is_train"] = False
        else:
            self.train_data[sentence_id] = True
            self.predictions["predicted_data"][sentence_id][1]["is_test"] = False
            self.predictions["predicted_data"][sentence_id][1]["is_train"] = True

        # if this example is a part of the test set, move it there
        if sentence_id in self.curr_test_query_ids and label is not None:
            self.move_to_test(sentence_id)

        if len(self.curr_query) == 0:
            # reset all query once everything is labeled
            self.experiment_stats["batch_end_time"].append(time.time())
            self.curr_query = None
            self.curr_query_predictions = None
            self.curr_test_query_ids = None
    
    def update_examples(self, sentence_ids: List[int], labels: List[List[str]]):
        for sentence_id, label in zip(sentence_ids, labels):
            self.update_example(sentence_id, label)

    def move_to_test(self, sentence_id: int):
        if sentence_id in self.train_data and sentence_id not in self.test_data:
            self.test_data[sentence_id] = True
            self.predictions["predicted_data"][sentence_id][1]["is_test"] = True
            self.predictions["predicted_data"][sentence_id][1]["is_train"] = False
            del self.train_data[sentence_id]
    
    def convert_word_ranges_to_labels(self, example: List[str], ranges: List[Tuple[int, int]]) -> List[str]:
        output = ['O' for _ in range(len(example))]
        if ranges is None:
            return None

        for (start, end) in ranges:
            if output[start] == 'O':
                # for merging issues, only overwrite negative labels
                # with start, else continue
                output[start] = f'B-{self.ner_class}'
            inside_tag = f'I-{self.ner_class}'
            for c_i in range(start + 1, end):
                output[c_i] = inside_tag
        return output
    
    def explain_labels(
        self,
        example: List[str],
        seq_label: List[str],
    ) -> Tuple[List[Tuple[int, int]], List[str]]:
        '''
        Convert a label to a list of word ranges and entities

        word_range[i] = (start: int, end: int) with end exclusive
        entities[i] = str, the entity corresponding to word_range[i]
        '''
        ranges : list = []
        entities: list = []
        range_start : int = None
        seq_label = [] if seq_label is None else seq_label
        for i, label in enumerate(seq_label):
            if (label == 'O' or i == len(seq_label) - 1) and range_start is not None:
                    ranges.append(
                        (
                            range_start,
                            i,
                        )
                    )
                    entities.append(' '.join(example[range_start : i]))
                    range_start = None
            elif label.startswith('B'):
                if range_start is not None:
                    ranges.append(
                        (
                            range_start,
                            i,
                        )
                    )
                    entities.append(' '.join(example[range_start : i]))
                range_start = i
        
        simple_entities: list = [ner.utils.get_simple_entity(e, return_str=True) for e in entities]

        # filter out empty entities
        simple_entities = [e for e in simple_entities if len(e) > 0]
        return ranges, entities, simple_entities
    
    def save_state(self, state_dict: Dict[str, object], key: str):
        state_dict[key] = {"state":
        (
            self.test_data,
            self.train_data,
            self.unlabeled_corpus,
            self.curr_query,
            self.curr_query_predictions,
            self.curr_test_query_ids,
            self.predictions,
            self.model,
            self.configuration,
            self.curr_sample_size,
            self.sampling_strategy,
        ),
        }

        self.database.save_state(state_dict[key], "db_manager")
    
    def load_state(self, state_dict: Dict[str, object], key: str):
        (
            self.test_data,
            self.train_data,
            self.unlabeled_corpus,
            self.curr_query,
            self.curr_query_predictions,
            self.curr_test_query_ids,
            self.predictions,
            self.model,
            self.configuration,
            self.curr_sample_size,
            self.sampling_strategy,
        ) = state_dict[key]["state"]

        self.database.load_state(state_dict[key], "db_manager")

    def save_model(self, file_name: str):
        print("saving model")
        torch.save(
            self.model.state_dict(), 
            os.path.join(file_name),
        )
    
    def load_model(self, file_name: str):
        print("loading model")
        self.model.load_state_dict(torch.load(file_name))
    
    def conll_data(self, dir_name: str):
        print("saving conll model")
        utils.conllize_database(
            dir_name=dir_name,
            database=self.database.database,
            train_ids=self.train_data.keys(),
            test_ids=self.test_data.keys(),
        )
            