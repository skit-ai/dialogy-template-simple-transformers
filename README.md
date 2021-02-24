# \[[python_package_import_name]]

Made with Dialogy Template with Simple Transformers

## Features

1.  XLMRWorkflow uses "xlm-roberta-base" for both classification and ner tasks.
2.  Flask integration.
3.  Sentry integration.

## Directory Structure

| File                                      | Description                                                                  |
| ----------------------------------------- | ---------------------------------------------------------------------------- |
| **config**                                | A directory that contains `yaml` files.                                      |
| **data**                                  | Version controlled by `dvc`.                                                 |
| **data/0.0.0**                            | A directory that would contain these directories: datasets, metrics, models. |
| **\[[python_package_import_name]]/dev**   | Programs not required in production.                                         |
| **\[[python_package_import_name]]/src**   | Programs required in production, makes smaller Dockerfiles.                  |
| **\[[python_package_import_name]]/utils** | Programs that offer assitance in either dev or src belong here.              |
| **CHANGELOG.md**                          | Track changes in the code, datasets, etc.                                    |
| **Dockerfile**                            | Containerize the application for production use.                             |
| **LICENSE**                               | Depending on your usage choose the correct copy, don't keep the default!     |
| **Makefile**                              | Helps maintain hygiene before deploying code.                                |
| **pyproject.toml**                        | Track dependencies here. Also, this means you would be using poetry.         |
| **README.md**                             | This must ring a bell.                                                       |
| **uwsgi.ini**                             | Modify as per use.                                                           |

## Getting started

### Pre-requisites

Make sure you have `git`, `python` pre installed. Preferably runnning python within a virtual environment.
The project runs on `python==^3.8`.

### Boilerplate

To setup a project using this template, use the following to generate the scaffolding:

```shell
pip install dialogy
dialogy create [[python_package_import_name]] dialogy-template-simple-transformers
```

The above should initiate an interactive session. The questions here will address some project level
details of thse, `project_name` creates a directory which contains all the relevant scaffolding. This 
template also expects [poetry](https://python-poetry.org/docs/), here
 are the [installation steps](https://python-poetry.org/docs/#installation).

### Dependencies

```shell
cd [[python_package_import_name]]
poetry install
```

### Data versioning and management

We use [`dvc`](https://dvc.org/doc/install) for dataset and model versioning. 
s3 is the preferred remote to save project level data that are not fit for tracking via git.

The `poetry install` step takes care of dvc installation.

```shell
git init
dvc init
dvc remote add -d s3remote s3://bucket-name/path/to/dir
poetry run dialogy data --version=0.0.0
dvc add data
```

This will create a data directory with the following structure:

```shell
data
+---0.0.0
    +---classification
    |   +---datasets
    |   +---metrics
    |   +---models
    +---ner
        +---datasets
        +---metrics
        +---models
```

It is evident that this template concerns itself with only `classification` and `ner` tasks.
You'd typically move your datasets into the datasets directory. The dataset should be split before hand into `train.csv` and `test.csv`
which is the expected naming convention.

### Training

Before training, your directory tree should look like:

```shell
data
+---0.0.0
    +---classification
    |   +---datasets
    |   |   +---train.csv
    |   |   +---test.csv
    |   +---metrics
    |   +---models
    +---ner
        +---datasets
        |   +---train.csv
        |   +---test.csv
        +---metrics
        +---models
```

In case you only need to train both classifier and ner models? run:

```shell
poetry run dialogy train [--version=<version>]
```

... need to train only `classifier`? run:

```shell
poetry run dialogy train classifier [--version=<version>]
```

... need to train only `ner`? run:

```shell
poetry run dialogy train ner [--version=<version>]
```

Once training is complete, you can expect model dir to be populated:

```shell
data
+---0.0.0
    +---classification
    |   +---datasets
    |   |   +---train.csv
    |   |   +---test.csv
    |   +---metrics
    |   +---models
    |       +---config.json
    |       +---eval_results.txt
    |       +---labelencoder.pkl
    |       +---model_args.json
    |       +---pytorch_model.bin
    |       +---sentencepiece.bpe.model
    |       +---special_tokens_map.json
    |       +---tokenizer_config.json
    |       +---training_args.bin
    |       +---training_progress_scores.csv
    +---ner
        +---datasets
        |   +---train.csv
        |   +---test.csv
        +---metrics
        +---models
            +---config.json
            +---entity_label_list.pkl
            +---eval_results.txt
            +---model_args.json
            +---pytorch_model.bin
            +---sentencepiece.bpe.model
            +---special_tokens_map.json
            +---tokenizer_config.json
            +---training_args.bin
```

### Evaluation

To evaluate replace the above commands with test instead of train like so:

```shell
poetry run dialogy test [--version=<version>]
poetry run dialogy test classifier [--version=<version>]
poetry run dialogy test ner [--version=<version>]
```

(If version is not provided, the default **0.0.0** is used from the _config.yaml_.)

Once tests are run, expect your directory to have a reports.csv. Not logging the models directory since it would take too much space and convey so little.

```shell
data
+---0.0.0
    +---classification
    |   +---datasets
    |   |   +---train.csv
    |   |   +---test.csv
    |   +---metrics
    |       +---report.csv
    |   <---models
    +---ner
        +---datasets
        |   +---train.csv
        |   +---test.csv
        +---metrics
            +---report.csv
        <---models
```

You may see it only in one directory depending on the test command arguments provided.

### Interactive Session

To run your models to see how they perform on live inputs, use the following command:

```shell
poetry run dialogy repl
```

This prints a set of expected input formats, **if nothing matches, it assumes the input to be plain-text!**
**Make sure you press ESC then ENTER to submit**. 

> The interface accepts multiline input and takes most people off-guard as they lie waiting for a response. The reason for putting in a multiline input is to offer convenience over pasting large json request bodies encountered in production and
> need to be tested locally for debugging.

### Releases

The project comes with an opinion on data management. The default branch (main/master) is expected to contain only the latest version of datasets and models.
The process creates a git tag with the semver so that you can checkout the tag for working on it in isolation to the rest of the project.

To initiate a release process, perform:

    poetry run dialogy release --version=<version>

## Commands

These are the available cli commands:

1.  poetry run dialogy train [--version=&lt;version&gt;]

    Routine for training both Classifier and NER sequntially.
    Provide a version and a model will be trained on a dataset of the same version.

    This script expects data/&lt;version> to be a directory where models, metrics
    and dataset are present.

2.  poetry run dialogy test [--version=&lt;version&gt;]

    Routine for testing both Classifier and NER sequentially.
    Provide a version to evaluate a trained model on an evaluation dataset.

3.  poetry run dialogy (train|test) (classification|ner) &lt;version>

    Same as the previous train and test commands with an exception of only one type of
    task (classification|ner) is picked.

4.  poetry run dialogy data [--version=&lt;version&gt;]

    This command creates a directory named &lt;version> under data.
    Helpful if only empty directory structures are needed.

5.  poetry run dialogy clone &lt;from_version> &lt;to_version>

    This command copies a directory from another under data.
    Helpful if only directory structures and their data should be copied.

6.  poetry run dialogy repl [--version=&lt;version&gt;]

    This command starts up an interactive terminal to dump json or plain text
    and interact with the trained models.

7.  poetry run dialogy release &lt;version>

    This command syncs dvc and git data, produces a tag on the repo and manages remote state.

```shell
Usage:
  __init__.py (train|test|repl) [--version=<version>]
  __init__.py (train|test) (classification|ner) [--version=<version>]
  __init__.py data --version=<version> [--force]
  __init__.py clone <from_version> <to_version> [--force]
  __init__.py release --version=<version>
  __init__.py (-h | --help)

Options:
    <from_version>          The source data directory; models, datasets, metrics will be copied from here.
    <to_version>            The destination data directory; models, datasets, metrics will be copied here.
    --version=<version>     The version of the dataset, model, metrics to use.
    --force                 Pass this flag to overwrite existing directories.
    -h --help               Show this screen.
```

## APIs

These are the APIs which are being used. Some of these are not needed in production.

1.  Health check - To check if the service is running.

    ```python
    @app.route("/", methods=["GET"])
    def health_check():
        return jsonify(
            status="ok",
            response={"message": "Server is up."},
        )
    ```

2.  Predict - This is the main production API.

    ```python
    @app.route("/predict/<lang>/<project_name>/", methods=["POST"])
    ```

## Customization

The best place to setup custom code is the `src` dir. The existing `workflow` would
usually be modified to have api level changes. The API itself can be modified via `api/endpoints.py`.

To modify configuration edit `config/config.yaml`.

## Upcoming Features

-   Stockholm
-   Data conflict
-   Data summarization
-   Visualization and interpretability
