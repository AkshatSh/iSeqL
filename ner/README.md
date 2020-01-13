# iSeqL NER Library

This is the library that holds all the algorithms and structures for neural entity recognition and active learning.

## File Structure

TODO

## Models Supported

WIP

* `elmo_bilstm_crf`
* `bilstm_crf`
* `dictionary`
* `phrase_dictionary`

## Active Learning

Algorithms supported

* `Random`
* `Uncertainty`
* `kNN`

```

Parameters:
    Iteration: Number of times to do active learning
    Max Epochs: Maximum number of epochs to run the algorithm
    Model: The model type to use
    Sample Size (K) : How many points to sample at once.

    ...Normal model and training hyper parameters.... 

Dependent Files:
    Unlabeled Corpus: This holds all the unlabeled data that the model will evaluate against
    Oracle Abstraction: The interface for an oracle that the model interacts with
        SimulatedOracle: An oracle that holds all the true labels and can return
        the correct label for a query
    Corpus Construction: To train on a data loader, this will convert the labeled and unlabled data
    into a torch dataset.

    Active Learining: A series of heuristics that given a model and an unlabeled corpus, can determine a ranking for the the queries in the unlabaled corpus. The higher the rank the more likely the model will ask that query to be labeled.

Generic Active Learning Algorithm:

    Start with K randomly sampled points
    Query user to get labels

    for each iteration:
        Train N epochs and find the best performance on the validation set, with current data

        predict model output on unlabled data

        sample K using sample strategy from unlabeled data

        agument training data with the new samples

        repeat

Things to Graph with Tensor Board:
    For each iteration:
        Valid Performance (f1, f1 per class)
        Train Performance (f1, f1 per class)

```

## To run bench marks

```
$ python train_conll.py --num_epochs e --load --train_bi_lstm
```

```
$ tensorboard --logdir=tensor_logs/
```

## Datasets

* Conll2003: (Person, Location, Organization, Misc)

## Related Papers

* [BERT](https://arxiv.org/pdf/1810.04805.pdf)
* [ELMo (Deep Contextualized Word Representations)](https://arxiv.org/pdf/1802.05365.pdf)
* [Semi Supervised Sequence Modeling with Cross View Training](https://arxiv.org/pdf/1809.08370.pdf)
* [Learning Named Entity Tagger using Domain-Specific Dictionary](https://arxiv.org/pdf/1809.03599.pdf)
* [SciIE](http://nlp.cs.washington.edu/sciIE/)
* [Deep Active Learning for Named Entity Recognition](https://arxiv.org/pdf/1707.05928.pdf)
  * [OpenReview](https://openreview.net/forum?id=ry018WZAZ)
