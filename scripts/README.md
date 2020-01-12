# Scripts

This folder contains a series of scripts useful for data preperation, repository set up, and analysis. The descriptions of each of them are available below:

## Set up

* `setup_nltk.py`: installs the necessary `NLTK` components for various parts of the NLP pipeline to run


## Data Preperation

* `prepare_cadec.py`: processes the `CADEC` dataset into BIO formatted data, data is set into a CSV file
* `prepare_conll_csv.py`: proccesses the `CoNLL` NER english dataset into a CSV, it is already BIO formatted
* `prepare_processed_scie.py`: `SCIERC` is a dataset collection of annoted AI paper abstracts, this script processes the NER information in this dataset and creates a BIO formated CSV training file for the pipeline, this uses the train test split defined by `SCIERC` authors
* `prepare_scie.py`: this uses the raw `SCIERC` data (without author splits), and creates a BIO formatted CSV.

## Analysis

* `experiment_analyzer.py`: loads user session data from the server and processes the learned models against a held out set, various metrics are reported and allows a model / user annotated data to be analyzed.