import os
import sys

# Repository data files, if this flag is set to true, repository assumes
# access to the data files and uses them in test suite
HAS_LOCAL_DATA_FILES: bool = False

# Special Tokens 
UNKNOWN_TOKEN = '<UNK>'
START_TOKEN = '<START>'
END_TOKEN = '<END>'
PAD_TOKEN = '<PAD>'
SPECIAL_TOKENS = [UNKNOWN_TOKEN, START_TOKEN, END_TOKEN, PAD_TOKEN]

# Conll2003 Dataset Paths
CONLL2003_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'conll2003', 'en/')
CONLL2003_TRAIN = os.path.join(CONLL2003_DIR, 'train.txt')
CONLL2003_TEST = os.path.join(CONLL2003_DIR, 'test.txt')
CONLL2003_VALID = os.path.join(CONLL2003_DIR, 'valid.txt')

# Conll2003 Input Keys
CONLL2003_WORD = 'word'
CONLL2003_POS = 'pos'
CONLL2003_SYNCHUNK = 'synchunk'

# SAVED_DATA_FILES
SAVED_DIR = os.path.join(os.path.dirname(__file__), 'saved/')
SAVED_CONLL_VOCAB = os.path.join(os.path.dirname(__file__), 'saved/', 'conll_vocab.pkl')
SAVED_CONLL_DATALOADER = os.path.join(os.path.dirname(__file__), 'saved/', 'conll_train_dataloader.pkl')
SAVED_CONLL_VALID_DATALOADER = os.path.join(os.path.dirname(__file__), 'saved/', 'conll_valid_dataloader.pkl')
SAVED_CONLL_CATEGORIES = os.path.join(os.path.dirname(__file__), 'saved/', 'conll_categories.pkl')

# SCIERC Dataset Path
SCIERC_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'sciie', 'raw_data/')
SCIERC_CONLL_DATASET = os.path.join(os.path.dirname(__file__), '..', 'data', 'sciie', 'scierc_conll.txt')
SCIERC_CONLL_VALID_DATASET = os.path.join(os.path.dirname(__file__), '..', 'data', 'sciie', 'scierc_valid_conll.txt')

# SCIERC Processed Dataset Conll 
SCIERC_CONLL_PROCESSED_TRAIN_DATASET = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'sciie',
    'scierc_processed_train_conll.txt',
)

SCIERC_CONLL_PROCESSED_VALID_DATASET = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'sciie',
    'scierc_processed_valid_conll.txt',
)

SCIERC_CONLL_PROCESSED_TEST_DATASET = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'sciie',
    'scierc_processed_test_conll.txt',
)

# SCIERC Processed dataset path
SCIERC_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'sciie', 'processed_data', 'json')
SCIERC_PROCESSED_TRAIN = os.path.join(SCIERC_PROCESSED_DIR, 'train.json')
SCIERC_PROCESSED_VALID = os.path.join(SCIERC_PROCESSED_DIR, 'dev.json')
SCIERC_PROCESSED_TEST = os.path.join(SCIERC_PROCESSED_DIR, 'test.json')

# SCIERC saved files
SCIERC_DATALOADER = os.path.join(os.path.dirname(__file__), 'saved/', 'scierc_dataloader.pkl')
SCIERC_VALID_DATALOADER = os.path.join(os.path.dirname(__file__), 'saved/', 'scierc_valid_dataloader.pkl')
SCIERC_VOCAB = os.path.join(os.path.dirname(__file__), 'saved/', 'scierc_vocab.pkl')
SCIERC_TAGS = os.path.join(os.path.dirname(__file__), 'saved/', 'scierc_tags.pkl')
SCIERC_PROCESSED_VOCAB = os.path.join(os.path.dirname(__file__), 'saved/', 'scierc_vocab.pkl')
SCIERC_PROCESSED_TAGS = os.path.join(os.path.dirname(__file__), 'saved/', 'scierc_tags.pkl')


# SCIERC Processed saved files
SCIERC_PROCESSED_TRAIN_DATALOADER = os.path.join(os.path.dirname(__file__), 'saved/', 'scierc_train_processed_dataloader.pkl')
SCIERC_PROCESSED_VALID_DATALOADER = os.path.join(os.path.dirname(__file__), 'saved/', 'scierc_valid_processed_dataloader.pkl')
SCIERC_PROCESSED_TEST_DATALOADER = os.path.join(os.path.dirname(__file__), 'saved/', 'scierc_test_processed_dataloader.pkl')

# CADEC directory
CADEC_DIR = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'cadec',
)

CADEC_CONLL_TRAIN_DATASET = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'cadec',
    'cadec_train_conll.txt',
)

CADEC_CONLL_VALID_DATASET = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'cadec',
    'cadec_valid_conll.txt',
)

CADEC_CONLL_POST_TRAIN_DATASET = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'cadec',
    'cadec_train_post_conll.txt',
)

CADEC_CONLL_POST_VALID_DATASET = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'cadec',
    'cadec_valid_post_conll.txt',
)

CADEC_CONLL_TRAIN_FINAL = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'cadec',
    'cadec_train_final.txt',
)

CADEC_CONLL_VALID_FINAL = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'cadec',
    'cadec_valid_final.txt',
)

CADEC_CONLL_TEST_FINAL = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'cadec',
    'cadec_test_final.txt',
)

CADEC_TRAIN_POST_DATALOADER = os.path.join(os.path.dirname(__file__), 'saved/', 'cadec_train_post_dataloader.pkl')
CADEC_VALID_POST_DATALOADER = os.path.join(os.path.dirname(__file__), 'saved/', 'cadec_valid_post_dataloader.pkl')
CADEC_POST_VOCAB = os.path.join(os.path.dirname(__file__), 'saved/', 'cadec_post_vocab.pkl')
CADEC_POST_TAGS = os.path.join(os.path.dirname(__file__), 'saved/', 'cadec_post_tags.pkl')

CADEC_TRAIN_DATALOADER = os.path.join(os.path.dirname(__file__), 'saved/', 'cadec_train_dataloader.pkl')
CADEC_VALID_DATALOADER = os.path.join(os.path.dirname(__file__), 'saved/', 'cadec_valid_dataloader.pkl')
CADEC_VOCAB = os.path.join(os.path.dirname(__file__), 'saved/', 'cadec_vocab.pkl')
CADEC_TAGS = os.path.join(os.path.dirname(__file__), 'saved/', 'cadec_tags.pkl')

# Active Learning Sampling
ACTIVE_LEARNING_SAMPLE = "sample"
ACTIVE_LEARNING_TOP_K = "top_k"

# Active Learning Hueristic
ACTIVE_LEARNING_RANDOM_H = "random"
ACTIVE_LEARNING_UNCERTAINTY_H = "uncertain"
ACTIVE_LEARNING_KNN = 'KNN'

# SCIERC DATASET OPTIONS
SCIERC_USE_PREPROCESSED = True

# CADEC DATASET OPTIONS
CADEC_USE_POST_LEVEL = True

LOG_ANALYSIS_FILES = False
