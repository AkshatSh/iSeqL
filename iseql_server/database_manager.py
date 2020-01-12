# python imports
from typing import (
    List,
    Dict,
)

# library imports
import os
import sys
from enum import Enum
import csv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.tokenize import PunktSentenceTokenizer
import pickle

NLTK_SIA = SIA()
PST = PunktSentenceTokenizer()

# local imports
from ner import (
    vocab
)
import constants
from configurations.configuration import Configuration

SESSION_INFO = {
    "0": {
        'name': "Document Level Cadec",
        'classes': ['ADR', 'DRUG'],
        'configuration_file': 'default_data_configuration.json',
    },
    "1": {
        "name": "Sentence Level Cadec",
        "classes": ['ADR', 'DRUG'],
        'configuration_file': 'default_data_configuration.json',
    },
    "2":{
        "name": "Document Level CONLL2003",
        "classes": ['PER'],
        'configuration_file': 'default_data_configuration.json',
    },
    "3":{
        "name": "Sentence Level CONLL2003",
        "classes": ['PER'],
        'configuration_file': 'default_data_configuration.json',
    },
    "4": {
        "name": "CADEC with Labels",
        "classes": ['ADR', 'DRUG'],
        'configuration_file': 'labeled_data_configuration.json',
    },
    "5": {
        "name": "IDL CADEC Scenario",
        "classes": ['ADR'],
        'configuration_file': 'labeled_data_configuration.json',
    },
    "6": {
        "name": "ADR Test Set Creation",
        "classes": ['ADR'],
        'configuration_file': 'labeled_data_configuration.json',
    },
    "7": {
        "name": "Yelp Review Data",
        "classes": ['Service'],
        'configuration_file': 'yelp_data_configuration.json',
    },
}


class Context(Enum):
    '''
    An enum for the different context levels
    a dataset can be evaluated in
    '''
    SENTENCE_LEVEL = 1 # all necessary context is within a sentence (PER, ORG, etc.)
    DOCUMENT_LEVEL = 2 # all necessary context is at the document level (ADR)

    @staticmethod
    def sentence_split(document: str) -> List[List[str]]:
        output: list  = []
        sent_scores: list = []
        spans: list = []
        for sent in nltk.sent_tokenize(document):
            sent_scores_d: dict = NLTK_SIA.polarity_scores(sent)
            word_tokens: list = nltk.word_tokenize(sent)

            span: list = [len(output), len(output) + len(word_tokens)]
            spans.append(span)
            output.append(word_tokens)
            sent_scores.append(sent_scores_d)
    
        return output, spans, sent_scores
    
    @staticmethod
    def document_split(document: str) -> List[List[str]]:
        sent_split, spans, sent_scores = Context.sentence_split(document)
        output = []
        for sent in sent_split:
            output.extend(sent)

        return [output], [spans], [sent_scores]


'''
A database manager:
    - The data folder contains a folder for each session for the user
    - contains methods, to add a new session
    - each session is associated with a dataset name, and the raw data
    - the dataset contains:
        a sqllite_db of the labeled examples/unlabeled examples
        the saved and processed files for training

API:
    - set the current session (expensive, loads everything into memory)
    - load (load state from disk)
    - save (expensive, saves the state of everything to disk)
    - add label params: (sentence_id, label) (updates sentence_id to have the label)

'''
class DatabaseManager(object):
    def __init__(self):
        # map of sentence_id to (sentence, label)
        self.database = {}
        self.entire_database = {}
        self.ground_truth_database = {}
        self.sentence_data = {}

        # map of entry_id to List[sentence_ids]
        self.entry_to_sentences = {}

        self.vocab = None
    
    def set_session(self, session_id: int):
        self.configuration = Configuration(SESSION_INFO[f'{session_id}']['configuration_file'])
        self.session_dir = os.path.join(
            constants.DATA_DIR,
            f'{session_id}/',
        )

        self.raw_data_csv = os.path.join(
            constants.DATA_DIR,
            session_id,
            self.configuration.get_key('data_file'),
        )

        self.context = Context.DOCUMENT_LEVEL if self.configuration.get_key('context_level') == 'DOCUMENT' else Context.SENTENCE_LEVEL
        self.context_func = Context.sentence_split if self.context == Context.SENTENCE_LEVEL else Context.document_split
    
    def get_all_session_info(self):
        return SESSION_INFO
    
    def _filter_label(self, label: str, ner_class: str) -> str:
        if label is None:
            return None
        def _filter_single_label(token: str, ner_class) -> str:
            if (token.startswith('B') or token.startswith('I')) and token[2:] != ner_class:
                return 'O'
            return token
        label_tokens = label.split(' ')
        filtered_label = [_filter_single_label(tok, ner_class) for tok in label_tokens]
        return filtered_label
    
    def prepare_csv(self):
        if self.load():
            print("load database manager state")
            return
        
        # necessary fields from data
        row_info = self.configuration.get_key('data_schema/rows')
        row_types = self.configuration.get_key('data_schema/row_types')
        text_field_name = self.configuration.get_key('data_schema/text_field')
        id_field_name = self.configuration.get_key('data_schema/id_field')
        label_field_name = self.configuration.get_key('data_schema/label_field')

        has_header = self.configuration.get_key('data_schema/includes_header')
    
        sentence_counter = 0
        word_list = []
        with open(self.raw_data_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if has_header:
                    # skip over headers in the data when included
                    has_header = False
                    continue
                row_data = {}

                # parse the row into the proper row names
                # extract the necessary fields from the configuration
                for row_name, row_value, row_type in zip(row_info, row, row_types):
                    if row_type == 'int':
                        row_data[row_name] = int(row_value)
                    elif row_type == 'float':
                        row_data[row_name] = float(row_value)
                    else:
                        row_data[row_name] = row_value


                entry_id = int(row_data[id_field_name])
                text_data = row_data[text_field_name]
                label = row_data[label_field_name] if label_field_name is not None else None


                # include computed metrics of data
                sent_scores: dict = NLTK_SIA.polarity_scores(text_data)
                row_data['sent_negative_score'] = sent_scores['neg']
                row_data['sent_neutral_score'] = sent_scores['neu']
                row_data['sent_positive_score'] = sent_scores['pos']
                row_data['sent_compound_score'] = sent_scores['compound']

                self.entire_database[entry_id] = row_data

                self.entry_to_sentences[entry_id] = []
                context_split, spans, sent_scores = self.context_func(text_data)
                for i, (model_entry, sent_span, sent_sent_score) in enumerate(zip(context_split,spans, sent_scores)):
                    if len(model_entry) == 0:
                        print(f'Error processing id: {(entry_id, i)} with sentence: {model_entry}')
                        continue
                    self.entry_to_sentences[entry_id].append(sentence_counter)
                    self.database[sentence_counter] = (model_entry, None)
                    self.ground_truth_database[sentence_counter] = (model_entry, self._filter_label(label, 'ADR'))
                    self.sentence_data[sentence_counter] = (
                        # sentence ranges
                        sent_span,
                        # sentence sentiments
                        sent_sent_score,
                    )

                    sent_spans = PST.span_tokenize(' '.join(model_entry)) 


                    sentence_counter += 1
                    word_list.extend(model_entry)
        
        self.vocab = vocab.build_vocab(word_list)
        self.save()

    def prepare(self):
        if self.configuration.get_key('data_schema/type'):
            self.prepare_csv()
        else:
            raise Exception("Unknown format of data")
    
    def load(self) -> bool:
        if not os.path.exists(os.path.join(self.session_dir, "vocab.pkl")):
            return False
        with open(os.path.join(self.session_dir, "vocab.pkl"), 'rb') as f:
            self.vocab = pickle.load(f)
        with open(os.path.join(self.session_dir, "entry_to_sentences.pkl"), 'rb') as f:
            self.entry_to_sentences = pickle.load(f)
        with open(os.path.join(self.session_dir, "database.pkl"), 'rb') as f:
            self.database = pickle.load(f)
        try:
            with open(os.path.join(self.session_dir, "entire_database.pkl"), 'rb') as f:
                self.entire_database = pickle.load(f)
        except Exception as e:
            self.entire_database = []
        try:
            with open(os.path.join(self.session_dir, "sentence_data.pkl"), 'rb') as f:
                self.sentence_data = pickle.load(f)
        except Exception as e:
            self.sentence_data = []
        
        # only if dataset has a ground truth (majority of datasets will not have this)
        ground_truth_database_path = os.path.join(self.session_dir, "ground_truth_database.pkl")
        if os.path.exists(ground_truth_database_path):
            with open(ground_truth_database_path, 'rb') as f:
                self.ground_truth_database = pickle.load(f)
        
        return True
    
    def save(self) -> bool:
        with open(os.path.join(self.session_dir, "vocab.pkl"), 'wb') as f:
            pickle.dump(self.vocab, f)
        with open(os.path.join(self.session_dir, "entry_to_sentences.pkl"), 'wb') as f:
            pickle.dump(self.entry_to_sentences, f)
        with open(os.path.join(self.session_dir, "database.pkl"), 'wb') as f:
            pickle.dump(self.database, f)
        with open(os.path.join(self.session_dir, "ground_truth_database.pkl"), 'wb') as f:
            pickle.dump(self.ground_truth_database, f)
        with open(os.path.join(self.session_dir, "entire_database.pkl"), 'wb') as f:
            pickle.dump(self.entire_database, f)
        with open(os.path.join(self.session_dir, "sentence_data.pkl"), 'wb') as f:
            pickle.dump(self.sentence_data, f)
        return True
    
    def add_label(self, sentence_id: int, label: List[str]) -> bool:
        return False
    
    def save_state(self, state_dict: Dict[str, object], key: str):
        state_dict[key] = (
            self.database,
            self.ground_truth_database,
            self.entry_to_sentences,
            self.vocab,
            self.configuration,
            self.session_dir,
            self.raw_data_csv,
            self.context,
            self.context_func,
        )
    
    def load_state(self, state_dict: Dict[str, object], key: str):
        (
            self.database,
            self.ground_truth_database,
            self.entry_to_sentences,
            self.vocab,
            self.configuration,
            self.session_dir,
            self.raw_data_csv,
            self.context,
            self.context_func,
        ) = state_dict[key]