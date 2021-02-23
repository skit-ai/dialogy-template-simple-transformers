# Dialogy Template with Simple Transformers

## Features

1.  XLMRWorkflow uses "xlm-roberta-base" for both classification and ner tasks.
2.  Flask integration.
3.  Sentry integration.

## Getting started

To setup a project using this template, use the following to generate the scaffolding:

```shell
pip install dialogy
dialogy app simple-transformers 
```

The above should initiate an interactive session. The questions here will address some project level
details of thse, `project_name` creates a directory which contains all the relevant scaffolding. This 
template also expects [poetry](https://python-poetry.org/docs/), here
 are the [installation steps](https://python-poetry.org/docs/#installation).

```shell
cd $project_name
poetry install
```

To setup [`dvc`](https://dvc.org/doc/install) for dataset and model versioning:

```shell
dvc init
dvc remote add -d s3remote s3://bucket-name/path/to/dir
poetry run dialogy data
dvc add data
```

## Directory Structure

| File               | Description                                                                  |
| ------------------ | ---------------------------------------------------------------------------- |
| **config**         | A directory that contains `yaml` files.                                      |
| **data**           | Version controlled by `dvc`.                                                 |
| **data/0.0.0**     | A directory that would contain these directories: datasets, metrics, models. |
| **dev**            | Programs not required in production.                                         |
| **src**            | Programs required in production, makes smaller Dockerfiles.                  |
| **Dockerfile**     | Containerize the application for production use.                             |
| **LICENSE**        | Depending on your usage choose the correct copy, don't keep the default!     |
| **Makefile**       | Helps maintain hygiene before deploying code.                                |
| **pyproject.toml** | Track dependencies here. Also, this means you would be using poetry.         |
| **README.md**      | This must ring a bell.                                                       |
| **uwsgi.ini**      | Modify as per use.                                                           |

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
