"""
This module offers the cli utility to interact with this project.

1. poetry run dialogy train [--version=<version>] [--file-format=<file_format>]

    Routine for training both Classifier and NER sequentially.
    Provide a version and a model will be trained on a dataset of the same version.

    This script expects data/<version> to be a directory where models, metrics
    and dataset are present.

2. poetry run dialogy test [--version=<version>] [--file-format=<file_format>] [--file=<test_file>]

    Routine for testing both Classifier and NER sequentially.
    Provide a version to evaluate a trained model on an evaluation dataset.

3. poetry run dialogy (train|test) (classification|ner) <version>  [--file-format=<file_format>]

    Same as the previous train and test commands with an exception of only one type of
    task (classification|ner) is picked.

4. poetry run dialogy data [--version=<version>]

    This command creates a directory named <version> under data.
    Helpful if only empty directory structures are needed.

5. poetry run dialogy clone <from_version> <to_version>

    This command copies a directory from another under data.
    Helpful if only directory structures and their data should be copied.

6. poetry run dialogy repl [--version=<version>]

    This command starts up an interactive terminal to dump json or plain text
    and interact with the trained models.

7. poetry run dialogy release <version>

    This command syncs dvc and git data, produces a tag on the repo and manages remote state.


Usage:
  __init__.py (train|test) [--version=<version>] [--file-format=<file_format>]
  __init__.py (train|test) (classification|ner) [--version=<version>] [--file-format=<file_format>] [--file=<test_file>]
  __init__.py data --version=<version> [--force]
  __init__.py clone <from_version> <to_version> [--force]
  __init__.py release --version=<version>
  __init__.py repl
  __init__.py (-h | --help)

Options:
    <from_version>              The source data directory; models, datasets, metrics will be copied from here.
    <to_version>                The destination data directory; models, datasets, metrics will be copied here.
    --version=<version>         The version of the dataset, model, metrics to use.
    --file-format=<file_format> One of "csv" or "sqlite".
    --file=<test_file>          A file to be tested separately.
    --force                     Pass this flag to overwrite existing directories.
    -h --help                   Show this screen.
"""
import semver
from docopt import docopt
from simpletransformers import classification

from slu import constants as const
from slu.dev.dir_setup import copy_data_directory, create_data_directory
from slu.dev.evaluate import test_classifier, test_ner
from slu.dev.release import release
from slu.dev.repl import repl
from slu.dev.train import train_intent_classifier, train_ner_model
from slu.utils.config import YAMLLocalConfig, Config
from slu.utils.logger import log


CLIENT_CONFIGS = YAMLLocalConfig().generate()


def main() -> None:
    args = docopt(__doc__)
    version = args["--version"]
    force = args["--force"]

    config: Config = list(CLIENT_CONFIGS.values()).pop()
    classification_task = config.task_by_name(const.CLASSIFICATION)
    ner_task = config.task_by_name(const.NER)

    if version:
        semver.VersionInfo.parse(version)
        config.version = version
    else:
        version = config.version

    if args.get(const.CLASSIFICATION):
        file_format = args["--file-format"] or const.CSV
    elif args[const.NER]:
        file_format = args["--file-format"] or const.CSV
    else:
        file_format = const.CSV

    if args[const.TRAIN] and args.get(const.CLASSIFICATION) and classification_task.use:
        train_intent_classifier(config, file_format=file_format)
        return None
    else:
        log.warning("Config is not prepared for using classification model.")

    if args[const.TRAIN] and args.get(const.NER) and ner_task.use:
        train_ner_model(config, file_format=file_format)
        return None
    else:
        log.warning("Config is not prepared for using ner model.")

    if args[const.TEST] and args.get(const.CLASSIFICATION) and classification_task.use:
        test_classifier(config, file_format=file_format, custom_file=args["--file"])
        return None
    else:
        log.warning("Config is not prepared for using classification model.")

    if args[const.TEST] and args.get(const.NER) and ner_task.use:
        test_ner(config, file_format=file_format, custom_file=args["--file"])
        return None
    else:
        log.warning("Config is not prepared for using ner model.")

    if args[const.TRAIN]:
        if classification_task.use:
            train_intent_classifier(config, file_format=file_format)
        else:
            log.warning("Config is not prepared for using classification model.")

        if ner_task.use:
            train_ner_model(config, file_format=file_format)
            return None
        else:
            log.warning("Config is not prepared for using ner model.")

    if args[const.TEST]:
        if classification_task.use:
            test_classifier(config, file_format=file_format, custom_file=args["--file"])
        else:
            log.warning("Config is not prepared for using classification model.")

        if ner_task.use:
            test_ner(config, file_format=file_format, custom_file=args["--file"])
        else:
            log.warning("Config is not prepared for using ner model.")

    if args[const.DATA]:
        create_data_directory(version, force=force)

    if args[const.CLONE]:
        copy_data_directory(
            copy_from=args["<from_version>"], copy_to=args["<to_version>"], force=force
        )

    if args[const.REPL]:
        repl()

    if args[const.RELEASE]:
        release(version)
