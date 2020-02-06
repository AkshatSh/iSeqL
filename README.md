# iSeqL: Interactive Sequence Learning

Repository for ACM IUI'20 Paper *iSeqL: Interactive Sequence Learning*

[![Travis Build](https://travis-ci.com/AkshatSh/iSeqL.svg?branch=master)](https://travis-ci.com/AkshatSh/iSeqL)
[![codecov](https://codecov.io/gh/AkshatSh/iSeqL/branch/master/graph/badge.svg)](https://codecov.io/gh/AkshatSh/iSeqL)

## Paper

[Shrivastava and Heer (2020): iSeqL: Interactive Sequence Learning](paper.pdf)

### Citation

TODO


## Documentation

This repository contains everything needed for `iSeqL` and the experiments run with it including the `NER` library, the server, and the UI, along with any unit tests, and scripts used. More detail on each of them is available at the README's in the corresponding folders.

* [iSeqL UI](iseql_front_end/README.md)
* [iSeqL Server](iseql_server/README.md)
* [iSeqL Server Configurations](iseql_server/configurations/README.md)
* [NER Library](ner/README.md)
* [Scripts](scripts/README.md)
* [Unit Tests for NER](tests/README.md)

## Set Up

To set up the python code create a python3 environment with the following:

```bash

# create a virtual environment
$ python3 -m venv env

# activate environment
$ source env/bin/activate

# install all requirements
$ pip install -r requirements.txt

# if using python 3.7.x
$ pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0-py3-none-any.whl
```

If you add a new package you will have to update the requirements.txt with the following command:

```bash

# add new packages
$ pip freeze > requirements.txt
```

And if you want to deactivate the virtual environment

```bash

# decativate the virtual env
$ deactivate
```
