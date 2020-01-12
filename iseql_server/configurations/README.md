# Configurations

Configurations are so users can specify how to handle different datasets, configurations for active learning, data processing settings. There is a default scheme provided to the user, so they can choose to leave fields as default values, and power users can tweak settings as they see need be. Configurations are provided for two things: `data` and `active_learning`, the schema is specified below:


## Data Config

Default Config

```json
{
    "default_schema": null,
    "data_file": "data.csv",
    "context_level": "DOCUMENT",
    "data_schema": {
        "type": "csv",
        "rows": ["id", "sent"],
        "row_types": ["int", "str"],
        "text_field": "sent",
        "id_field": "id",
        "label_field": null,
        "label_ner_class": null,
        "includes_header": false
    }
}
```

### Data Field Definitions

* `default_schema`: the default schema to inherit from, this copies values from this schema into the underspecified ones in the current schema, most users should set this to `default_data_configuration.json`
* `data_file`: the file that contains all the data that the user wants to explore
* `context_level`: `DOCUMENT` or `SENTENCE`, should iSeqL operate at the document level or should it operate at the sentence level. Unless each document is multiple paragraphs, we tend to use `DOCUMENT` level.
* `data_schema`: the configuration for the data schema itself.
    * `type`: `CSV`, current `CSV` is the only supported type, but this could easily be extended to support other data types as well.
    * `rows`: the rows that should be stored for analysis
    * `row_types`: the types associated with each row (`int`, `str`, `float`)
    * `text_field`: the field that corresponds to the text field in the datset
    * `id_field`: the field that corresponds to the unique id of each row
    * `label_field`: the field that contains the correct BIO labels if there are any (could be set to `null`)
    * `label_ner_class`: the name of the `NER` class that the user wants to analyze (e.g. `ADR` for adverse drug reaction, or `Person` for identifying people etc.), user should decide an informative name so they can use this for analysis.
    * `includes_header`: Does the csv contain a header line.


## Active Learning Config

```json
{
    "default_schema": null,
    "model_schema": {
        "model_type": "elmo_bilstm_crf",
        "embedding_dim": 1024,
        "hidden_dim": 1000
    },
    "active_learning_sampling_rate": [20, 20, 20, 20, 20],
    "trainer_params": {
        "learning_rate": 0.01,
        "weight_decay": 1e-4,
        "momentum": 0.0,
        "optimizer_type": "SGD",
        "batch_size": 1,
        "num_workers": 2,
        "num_epochs": 5
    },
    "active_heuristic": "RANDOM",
    "sampling_strategy": "top_k",
    "comparison_metric": "train_f1_avg",
    "test_set_split": 0.2
}
```

### AL Field Definitions

* `default_schema`: the default schema to inherit from, this copies values from this schema into the underspecified ones in the current schema, most users should set this to `default_data_configuration.json`
* `model_schema`: specifies any model parameters
    * `model_type`: uses the model type from the NER library (e.g. `elmo_bilstm_crf`, `bilstm_crf`, `dictionary`)
    * `embedding_dim`: the dimension of the embedding (`1024` for ELMo)
    * `hidden_dim`: the number of hidden dimensions being used for the `bilstm` component
* `active_learning_sampling_rate`: a list of `int` specifying at each step what the corresponding sample size should be.
* `trainer_parms`: a series of parameters for the optimizer
    * `learning_rate`: the learning rate for the optimizer being used
    * `weight_decay`: regualizer for the weight decay
    * `momentum`: momentum for the optimizer
    * `batch_size`: number of instances to process at each iteration
    * `num_workers`: how many threads should be used for training
    * `num_epochs`: number of epochs (iterations over the dataset), should be used for training
* `active_learning_heuristic`: Which active learning heuristic is being used for selecting the next sample (`RANDOM`, `UNCERTAINTY`, `KNN`)
* `sampling_strategy`: How should samples be selected once the heuristic scores them (`top_k`: select the top k, `sample`: do a weighted sample with the scores from the heuristic)
* `comparision_metric`: What metric should be used to determine which model is the best
* `test_set_split`: What percentage of samples should be going to the test set (default: `0.1` or 10%)