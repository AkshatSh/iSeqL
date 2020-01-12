import os 
import random
from io import TextIOWrapper
import torch
import csv
from torch import nn
from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords 
import spacy
STOP_WORDS = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')

from ner.models.cached_embedder import CachedEmbedder
from ner.models.cached_bilstm_crf import CachedBiLSTMCRF
from ner.models.elmo import FrozenELMo

try: # pragma: no cover
    # script
    import constants

    # modeling imports
    import models.bilstm_crf as bilstm_crf
    # from models.advanced_tutorial import BiLSTM_CRF
    import models.elmo_bilstm_crf as elmo
    import models.crf as crf
    import models.dictionary_model as dictionary_model
except: # pragma: no cover
    # module
    from . import constants

    # modeling imports
    from .models import bilstm_crf as bilstm_crf
    # from .models.advanced_tutorial import BiLSTM_CRF
    from .models import elmo_bilstm_crf as elmo
    from .models import crf
    from .models import dictionary_model

def get_or_create_file(file_name):
    '''
    Given a file name create any dependent directories
    '''
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name) and dir_name:
        os.makedirs(dir_name)
    return open(file_name, 'w')


def compute_f1(predicted, actual, categories):
    '''
    Given a predicted tag sequence and the actual tag sequence 
    computes the true positive, false positive and false negative
    for each class
    '''
    epoch_overall_inc = sum(actual[i] != predicted[i] for i in range(len(predicted)))
    epoch_overall_total = len(actual)
    output = {}
    for cat in categories:
        tp = sum( (actual[i] == cat) and (predicted[i] == cat)  for i in range(len(predicted)))
        fp = sum( (actual[i] == cat) and (predicted[i] != cat)  for i in range(len(predicted)))
        fn = sum( (actual[i] != cat) and (predicted[i] == cat)  for i in range(len(predicted)))
        output[cat] = (tp, fp, fn)
    return output

def combine_f1(old_f1, new_f1):
    out = {}
    for key in old_f1:
        tp, fp, fn = old_f1[key]
        tp_2, fp_2, fn_2 = new_f1[key]
        out[key] = (tp + tp_2, fp + fp_2, fn + fn_2)
    
    return out


def get_precision_recall_f1(f1_data):
    out = {}
    for cat in f1_data:
        tp, fp, fn = f1_data[cat]
        precession = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * (precession * recall) / (precession + recall) if (precession + recall) != 0 else 0
        out[cat] = {
            'precision': precession,
            'recall': recall,
            'f1': f1,
        }
    return out

def remove_bio_input(input):
    '''
    remove bio encoding from one specific tag
    '''
    index = input.find('-')
    return input[index + 1:] if index != -1 else input

def remove_bio(inputs):
    '''
    Remvoe bio encoding from output sequence
    '''
    return [remove_bio_input(tag) for tag in inputs]

def smart_log(
    sentence: list,
    predicted: list,
    expected: list,
    log_file: TextIOWrapper,
    uncertain: torch.Tensor,
    csv_writer: Optional[csv.writer],
) -> None:
    header = '''
<----------------------------------->
Sentence: {}
Expected: {}
Predicted: {}
Uncertain: {}

'''.format(sentence, expected, predicted, uncertain)
    log_file.write(header)
    for i in range(len(predicted)):
        if csv_writer is not None: # pragma: no cover
            csv_writer.writerow(
                [sentence[i], predicted[i], expected[i]]
            )
        if predicted[i] != expected[i]:
            log_file.write(
                "{}, {}, {}\n".format(sentence[i], predicted[i], expected[i])
            )

def analyze_predictions(
    model: nn.Module,
    data_loader: object,
    vocab:object,
    tag_vocab: object,
    log_file: TextIOWrapper,
    csv_file: Optional[TextIOWrapper],
    device: str = 'cpu',
):
    model.eval()
    log_file.write(
        "Sentence, Predicted, Expected\n"
    )

    csv_writer = None
    if csv_file is not None: # pragma: no cover
        csv_writer = csv.writer(csv_file, delimiter=',')

        csv_writer.writerow(['Sentence', 'Predicted', 'Expected'])

    with torch.no_grad():
        for i, (s_ids, x, x_chars, y, _) in enumerate(tqdm(data_loader)):
            x, x_chars = x.to(device), x_chars.to(device)
            out = model(x, x_chars)
            uncertain = model.compute_uncertainty(x, x_chars, s_ids)

            sentence = [vocab.get_word(int(index)) for index in x[0]]
            predicted = remove_bio([tag_vocab.get_word(int(index)) for index in out[0]])
            expected = remove_bio([tag_vocab.get_word(int(index)) for index in y[0].data.tolist()])
            smart_log(
                sentence,
                predicted,
                expected,
                log_file,
                uncertain,
                csv_writer,
            )
        
    model.train()

def estimate_f1(
    model: nn.Module,
    data_loader: object,
    tag_vocab: object,
    threshold: int,
    device: str = 'cpu',
):
    return compute_f1_dataloader(model, data_loader, tag_vocab, threshold, device)

def compute_labels(model, data_loader, tag_vocab, verbose: bool = False, device: str ='cpu') -> List:
    model.eval()
    results = []

    if verbose:
        data_loader = tqdm(data_loader)

    with torch.no_grad():
        for batch_i, model_args in enumerate(data_loader):
            s_ids = model_args[0] if len(model_args) == 3 else None
            x, x_chars = model_args if s_ids is None else model_args[1:]
            x, x_chars = x.to(device), x_chars.to(device)
            preds = model(x, x_chars, s_ids)
            for pred in preds:
                # decode everything in the batch
                predicted = [tag_vocab.get_word(int(index)) for index in pred]
                results.append(predicted)
    return results


def compute_f1_dataloader(model, data_loader, tag_vocab, threshold=None, device: str ='cpu'):
    '''
    Compute f1 for model in a data loader
    '''
    model.eval()
    
    correct = 0
    total = 0

    f1 = {}

    bio_removed_tags = remove_bio(tag_vocab.get_all())

    # iterate over epochs
    with torch.no_grad():
        for batch_i, (s_ids, x, x_chars, y, weights) in enumerate(data_loader):
            s_ids, x, x_chars = s_ids.to(device), x.to(device), x_chars.to(device)
            if threshold is not None and (batch_i > threshold):
                break
            model.zero_grad()
            out = model(x, x_chars, s_ids)

            predicted = remove_bio([tag_vocab.get_word(int(index)) for index in out[0]])
            expected = remove_bio([tag_vocab.get_word(int(index)) for index in y[0].data.tolist()])

            total += len(predicted)
            correct += sum([predicted[i] == expected[i] for i in range(len(predicted))])

            f1_temp = compute_f1(predicted, expected, bio_removed_tags)

            if len(f1) == 0:
                f1 = f1_temp
            else:
                f1 = combine_f1(f1, f1_temp)
            # break
    
    print('correct: {} total: {} accuracy: {}'.format(correct, total, correct / total if total > 0 else 0))
    model.train()
    return (get_precision_recall_f1(f1), correct / total if total > 0 else 0)

def log_metrics(logger, f1_data, prefix, step_count):
    '''
    Given tag f1 data and a logger, log all the information
    '''
    for tag in f1_data:
        if tag not in constants.SPECIAL_TOKENS:
            for metric in f1_data[tag]:
                logger.scalar_summary(
                    "{}/{}/{}".format(prefix, tag, metric), 
                    f1_data[tag][metric], 
                    step_count,
                )


def sample_array(array, percent):
    '''
    Given an array and a float percent (between 0.0 and 1.0) returns a shuffled part 
    of the array that has the new size floor[len(array) * percent]
    '''
    # random.seed(1)
    new_size = int(len(array) * percent)
    for i in range(new_size):
        ind = random.randint(0, new_size - 1)
        temp = array[i]
        array[i] = array[ind]
        array[ind] = temp
    
    return array[:new_size]

def compute_avg_f1(f1_data):
    '''
    Given the f1_data object compute the average f1 score
    '''
    if len(f1_data) == 0:
        return 0
    scores = []
    for key in f1_data:
        if key not in constants.SPECIAL_TOKENS:
            scores.append(f1_data[key]['f1'])
    avg = sum(scores) / len(scores)
    return avg

def print_f1_summary(f1_data: dict, title="F1 Summary"):
    '''
    Print an overview of the f1 object
    '''
    print("{}{}{}".format(
        '=' * 30,
        title,
        '=' * 30,
    ))
    for key in f1_data:
        if key not in constants.SPECIAL_TOKENS:
            summary = "{CLASS} | F1: {F1} | Precession : {P} | Recall : {R}".format(
                CLASS=key,
                F1=f1_data[key]['f1'],
                P=f1_data[key]['precision'],
                R=f1_data[key]['recall'],
            )
            print("\t{}".format(summary))
    print("=" * (30 * 2 + len(title)))

def build_model(
    model_type: str, # model_type="bilstm_crf"
    embedding_dim: int,
    hidden_dim: int,
    vocab: object,
    tag_vocab: object, #vocab object but avoiding circular dependency
    batch_size: int,
) -> torch.nn.Module:
    '''
    Given a configuration, constructs a model
    '''
    if model_type == "bilstm_crf":
        print('using bilstm_crf')
        return bilstm_crf.BiLSTM_CRF(vocab, tag_vocab, embedding_dim, hidden_dim, batch_size)
    elif model_type == "elmo_bilstm_crf":
        print('using elmo_bilstm_crf')
        return elmo.ELMo_BiLSTM_CRF(vocab, tag_vocab, hidden_dim, batch_size)
    elif model_type == "dictionary":
        print('using dictionary model')
        return dictionary_model.DictionaryModel(vocab, tag_vocab)
    elif model_type == "phrase_dictionary":
        print('using phrase dictionary')
        return dictionary_model.PhraseDictionaryModel(vocab, tag_vocab)
    elif model_type == "cached":
        cached_embedder = CachedEmbedder(
                embedder=FrozenELMo.instance(),
                embedding_dimensions=FrozenELMo.DIMENSIONS,
        )
        model = CachedBiLSTMCRF(
            vocab=vocab,
            tag_set=tag_vocab,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            embedder=cached_embedder,
        )
        return model

def get_simple_entity(
    entity: object,
    return_str: bool = False,
) -> List[str]:
    if type(entity) == list:
        entity = ' '.join(entity)
    entity = entity.strip()
    spacy_res = nlp(entity,  disable=['parser'])
    
    entity_res = []
    for ent_tok in spacy_res:
        if not ent_tok.is_stop and ent_tok.lemma_ not in STOP_WORDS and not ent_tok.is_punct:
            lemma = ent_tok.lemma_
            if lemma == "-PRON-":
                lemma = "<PRONOUNS>"
            entity_res.append(lemma)
    
    entity_res.sort()
    if return_str:
        return ' '.join(entity_res)
    return entity_res
    