"""
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
import argparse
from typing import Optional

from slu import constants as const
from slu.dev.dir_setup import copy_data_directory, create_data_directory
from slu.dev.test import test_classifier
from slu.dev.release import release
from slu.dev.train import train_intent_classifier
from slu.dev.repl import repl
from slu.utils.config import Config, YAMLLocalConfig
from slu.utils import logger


CLIENT_CONFIGS = YAMLLocalConfig().generate()


def build_dir_cli(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--version",
        default="0.0.1",
        help="The version of the dataset, model, metrics to use. Defaults to the latest version.",
    )
    return parser


def build_train_cli(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--file",
        help="A csv dataset containing utterances and labels.",
    )
    parser.add_argument("--lang", help="The language of the dataset.")
    parser.add_argument(
        "--project", help="The project scope to which the dataset belongs."
    )
    parser.add_argument(
        "--version", help="The dataset version, which will also be the model's version."
    )
    return parser


def build_test_cli(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--file", help="A csv dataset containing utterances and labels."
    )
    parser.add_argument("--lang", help="The language of the dataset.")
    parser.add_argument(
        "--project", help="The project scope to which the dataset belongs."
    )
    parser.add_argument(
        "--version",
        help="The dataset version, which will also be the report's version.",
    )
    return parser


def build_clone_cli(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--source", required=True, help="The version of the source data directory."
    )
    parser.add_argument(
        "--dest", required=True, help="The version of the destination data directory."
    )
    return parser


def build_release_cli(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--version",
        required=True,
        help="The version of the dataset, model, metrics to use. Defaults to the latest version.",
    )
    return parser


def build_repl_cli(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--lang", help="Run the models and pre-processing for the given language code."
    )
    return parser


def parse_commands(command_string: Optional[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    command_parsers = parser.add_subparsers(dest="command", help="Project utilities.")
    dir_cli_parser = command_parsers.add_parser(
        "dir-setup", help="Create base directory structure."
    )
    train_cli_parser = command_parsers.add_parser("train", help="Train a workflow.")
    test_cli_parser = command_parsers.add_parser("test", help="Test a workflow.")
    clone_cli_parser = command_parsers.add_parser(
        "clone", help="Clone a version of the data directory."
    )
    release_cli_parser = command_parsers.add_parser(
        "release", help="Release a version of the project."
    )
    repl_cli_parser = command_parsers.add_parser(
        "repl", help="Read Eval Print Loop for a trained workflow."
    )

    dir_cli_parser = build_dir_cli(dir_cli_parser)
    train_cli_parser = build_train_cli(train_cli_parser)
    test_cli_parser = build_test_cli(test_cli_parser)
    clone_cli_parser = build_clone_cli(clone_cli_parser)
    release_cli_parser = build_release_cli(release_cli_parser)
    repl_cli_parser = build_repl_cli(repl_cli_parser)

    command = command_string.split() if command_string else None
    return parser.parse_args(command)


def main(command_string: Optional[str] = None) -> None:
    args = parse_commands(command_string=command_string)
    if args.command == "dir-setup":
        create_data_directory(args)
    elif args.command == "train":
        train_intent_classifier(args)
    elif args.command == "test":
        test_classifier(args)
    elif args.command == "clone":
        copy_data_directory(args)
    elif args.command == "release":
        release(args)
    elif args.command == "repl":
        repl(args)
    else:
        raise ValueError("Unrecognized command: {}".format(args.command))
