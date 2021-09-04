# SLU

Made with Dialogy Template with Simple Transformers

## Features

1. XLMRWorkflow uses "xlm-roberta-base" for both classification and ner tasks.
2. Flask integration.
3. Sentry integration.

## Directory Structure

| File                                      | Description                                                                  |
| ----------------------------------------- | ---------------------------------------------------------------------------- |
| **config**                                | A directory that contains `yaml` files.                                      |
| **data**                                  | Version controlled by `dvc`.                                                 |
| **data/0.0.1**                            | A directory that would contain these directories: datasets, metrics, models. |
| **slu/dev**                               | Programs not required for development, might not be useful in production.    |
| **slu/src**                               | Houses the prediction API.                                                   |
| **slu/utils**                             | Programs that offer assitance in either dev or src belong here.              |
| **tests/**                                | Test cases for your project.                                                 |
| **CHANGELOG.md**                          | Track changes in the code, datasets, etc.                                    |
| **Dockerfile**                            | Containerize the application for production use.                             |
| **LICENSE**                               | Depending on your usage choose the correct copy, don't keep the default!     |
| **Makefile**                              | Helps maintain hygiene before deploying code.                                |
| **pyproject.toml**                        | Track dependencies here. Also, this means you would be using poetry.         |
| **README.md**                             | This must ring a bell.                                                       |
| **uwsgi.ini**                             | Modify as per use.                                                           |

## Getting started

Make sure you have `git`, `python==^3.8`, [`poetry`](https://python-poetry.org/docs/#installation) installed. Preferably within a virtual environment.

### 1. Boilerplate

To create a project using this template, run:

```shell
pip install dialogy
dialogy create hello-world
```

The questions here help:

- Populate your [`pyproject.toml`](https://python-poetry.org/docs/pyproject/) since we use [`poetry`](https://python-poetry.org/docs/) for managing dependencies.
- Create a repository and python package with the scaffolding you need.

### 2. Install

```shell
cd hello-world
poetry install
make lint
git init
git add .
git commit -m "add: initial commit."
```

**Please look at `"languages"` key in `config.yaml`. Update this with supported languages to prevent hiccups!**

### 3. Version control

We use [`dvc`](https://dvc.org/doc/install) for dataset and model versioning.
s3 is the preferred remote to save project level data that are not fit for tracking via git.

```shell
# from project root.
dvc init
dvc add data
dvc remote add -d myremote s3://bucket/path/to/some/dir
git add data.dvc
```

### 4. Project setup

The `poetry install` step takes care of dvc installation. You need to create a project on github, gitlab, bitbucket, etc.
set the remote. Once you are done with the installation, you can perform `slu -h`.

```shell
> slu -h
usage: slu [-h] {setup-dirs,split-data,combine-data,train,test,release,repl} ...

positional arguments:
  {setup-dirs,split-data,combine-data,train,test,release,repl}
                        Project utilities.
    setup-dirs          Create base directory structure.
    split-data          Split a dataset into train-test datasets for given ratio.
    combine-data        Combine datasets into a single file.
    train               Train a workflow.
    test                Test a workflow.
    release             Release a version of the project.
    repl                Read Eval Print Loop for a trained workflow.

optional arguments:
  -h, --help            show this help message and exit
```

### 5. Data setup

Let's start with dataset, model and report management command `slu setup-dirs --version=0.0.1`.

```shell
slu setup-dirs -h
usage: slu setup-dirs [-h] [--version VERSION]

optional arguments:
  -h, --help         show this help message and exit
  --version VERSION  The version of the dataset, model, metrics to use. Defaults to the latest version.
```

This creates a data directory with the following structure:

```shell
data
+---0.0.1
    +---classification
        +---datasets
        +---metrics
        +---models
```

### 6. Data Preparation

Assuming we have a labeled dataset, we are ready to execute the next command `slu split-data`,
this puts a `train.csv` and `test.csv` at a desired `--dest` or the project default places within
`data/0.0.1/classification/datasets`.

```shell
slu split-data -h
usage: slu split-data [-h] [--version VERSION] --file FILE (--train-size TRAIN_SIZE | --test-size TEST_SIZE)
                      [--stratify STRATIFY] [--dest DEST]

optional arguments:
  -h, --help            show this help message and exit
  --version VERSION     The version for dataset paths.
  --file FILE           A dataset to be split into train, test datasets.
  --train-size TRAIN_SIZE
                        The proportion of the dataset to include in the train split
  --test-size TEST_SIZE
                        The proportion of the dataset to include in the test split.
  --stratify STRATIFY   Data is split in a stratified fashion, using the class labels. Provide the column-name in
                        the dataset that contains class names.
  --dest DEST           The destination directory for the split data.
```

```shell
data
+---0.0.1
    +---classification
    +---datasets
    |   +---train.csv
    |   +---test.csv
    +---metrics
    +---models
```

### 7. Train

To train an classifier, we run `slu train`.

```shell
slu train -h
usage: slu train [-h] [--file FILE] [--lang LANG] [--project PROJECT] [--version VERSION]

optional arguments:
  -h, --help         show this help message and exit
  --file FILE        A csv dataset containing utterances and labels.
  --lang LANG        The language of the dataset.
  --project PROJECT  The project scope to which the dataset belongs.
  --version VERSION  The dataset version, which will also be the model's version.
```

Not providing the `--file` argument will pick a `train.csv` from `data/0.0.1/classification/datasets`.
Once the training is complete, you would notice the models would be populated:

```shell
data
+---0.0.1
    +---classification
    +---datasets
    |   +---train.csv
    |   +---test.csv
    +---metrics
    +---models
        +---config.json
        +---eval_results.txt
        +---labelencoder.pkl
        +---model_args.json
        +---pytorch_model.bin
        +---sentencepiece.bpe.model
        +---special_tokens_map.json
        +---tokenizer_config.json
        +---training_args.bin
        +---training_progress_scores.csv
```

### 8. Evaluation

We evaluate all the plugins in the workflow using `slu test --lang=LANG`.
Not providing the `--file` argument will pick a `test.csv` from `data/0.0.1/classification/datasets`.

```shell
slu test -h
usage: slu test [-h] [--file FILE] --lang LANG [--project PROJECT] [--version VERSION]

optional arguments:
  -h, --help         show this help message and exit
  --file FILE        A csv dataset containing utterances and labels.
  --lang LANG        The language of the dataset.
  --project PROJECT  The project scope to which the dataset belongs.
  --version VERSION  The dataset version, which will also be the report's version.
```

Reports are saved in the `data/0.0.1/classification/metrics` directory. We save:

1. A classification report that shows the f1-score for all the labels in the `test.csv` or `--file`.

2. A confusion matrix between a select intents.

3. A collection of all the data-points where the predictions don't match the ground truth.

### 9. Interact

To run your models to see how they perform on live inputs, you have two options:

1. `slu repl`

    ```shell
    slu repl -h
    usage: slu repl [-h] [--version VERSION] [--lang LANG]

    optional arguments:
    -h, --help         show this help message and exit
    --version VERSION  The version of the dataset, model, metrics to use. Defaults to the latest version.
    --lang LANG        Run the models and pre-processing for the given language code.
    ```

    The multi-line input catches people off-guard. `ESC` + `ENTER` to submit an input to the repl.

2. `task serve`

    This is a uwsgi server that provides the same interface as your production applications.

### 10. Releases

Once the model performance achieves a satisfactory metric, we want to release and persist the dataset, models and reports.
To do this, we meet the final command `slu release --version VERSION`.

```shell
slu release -h
usage: slu release [-h] --version VERSION

optional arguments:
  -h, --help         show this help message and exit
  --version VERSION  The version of the dataset, model, metrics to use. Defaults to the latest version.
```

This command takes care of the following acts:

1. Stages `data` dir for dvc.

2. Requires a changelog input.

3. Stages changes within CHANGELOG.md, data.dvc, config.yaml, pyproject.toml for content updates and version changes.

4. Creates a commit.

5. Creates a tag for the given `--version=VERSION`.

6. Pushes the data to dvc remote.

7. Pushes the code and tag to git remote.

## 11. Build

Finally, we are ready to build a Docker image for our service for production runs. We use Makefiles to ensure a bit of hygiene checks.
Run `make <image-name>` to check if the image builds in your local environment. If you have CI-CD enabled, that should do it for you.

## Config

The config manages paths for artifacts, arguments for models and rules for plugins.

```yaml
calibration: {}
languages:
- en
model_name: slu
slots:                      # Arbitrary slot filing rule to serve as an example.
  _cancel_:
    number_slot:
    - number
tasks:
  classification:
    alias: {}
    format: ''
    model_args:
      production:
        best_model_dir: data/0.0.1/classification/models
        dynamic_quantize: true
        eval_batch_size: 1
        max_seq_length: 128
        no_cache: true
        output_dir: data/0.0.1/classification/models
        reprocess_input_data: true
        silent: true
        thread_count: 1
        use_multiprocessing: false
      test:
        best_model_dir: data/0.0.1/classification/models
        output_dir: data/0.0.1/classification/models
        reprocess_input_data: true
        silent: true
      train:
        best_model_dir: data/0.0.1/classification/models
        early_stopping_consider_epochs: true
        early_stopping_delta: 0.01
        early_stopping_metric: eval_loss
        early_stopping_metric_minimize: true
        early_stopping_patience: 3
        eval_batch_size: 8
        evaluate_during_training_steps: 1080
        fp16: false
        num_train_epochs: 1
        output_dir: data/0.0.1/classification/models
        overwrite_output_dir: true
        reprocess_input_data: true
        save_eval_checkpoints: false
        save_model_every_epoch: false
        save_steps: -1
        use_early_stopping: true
    skip:                                   # Remove these intents from training data.
    - silence
    - audio_noisy
    threshold: 0.1
    use: true
version: 0.0.1
```

Model args help maintain the configuration of models in a single place, [here](https://simpletransformers.ai/docs/usage/#configuring-a-simple-transformers-model) is a full list, for classification or NER model configuration.

## APIs

These are the APIs which are being used.

1. Health check - To check if the service is running.

    ```python
    @app.route("/", methods=["GET"])
    def health_check():
        return jsonify(
            status="ok",
            response={"message": "Server is up."},
        )
    ```

2. Predict - This is the main production API.

    ```python
    @app.route("/predict/<lang>/slu/", methods=["POST"])
    ```
