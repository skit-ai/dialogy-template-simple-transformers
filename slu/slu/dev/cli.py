"""
[summary]
"""
import argparse
from typing import Optional

from slu.dev.dev import dev_workflow
from slu.dev.dir_setup import create_data_directory
from slu.dev.prompt_setup import setup_prompts, fill_nls_col
from slu.dev.repl import repl
from slu.dev.test import test_classifier
from slu.dev.train import create_data_splits, merge_datasets, train_intent_classifier


def build_split_data_cli(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--file", help="A dataset to be split into train, test datasets.", required=True
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--train-size",
        help="The proportion of the dataset to include in the train split",
        type=float,
    )
    group.add_argument(
        "--test-size",
        help="The proportion of the dataset to include in the test split.",
        type=float,
    )
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Data is split in a stratified fashion, using the class labels.",
    )
    parser.add_argument("--dest", help="The destination directory for the split data.")
    return parser


def build_data_combine_cli(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--out", help="The csv file.", required=True)
    parser.add_argument(
        "files", nargs="*", help="The path of the files to be combined into one."
    )
    return parser


def build_train_cli(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--file",
        help="The path of a csv dataset containing utterances and labels. If not provided, we look for files in data/classification/datasets.",
    )
    parser.add_argument("--lang", help="The language code to use for the dataset.")
    parser.add_argument(
        "--project", help="The project scope to which the dataset belongs."
    )
    parser.add_argument(
        "--epochs",
        help="The number of epochs to train the model for.",
        type=int,
    )
    return parser


def build_dev_cli(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--file",
        help="The path of a csv dataset containing utterances and labels. If not provided, we look for files in data/classification/datasets.",
    )
    parser.add_argument(
        "--lang", help="The language code to use for the dataset.", required=True
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="To calibrate model confidence scores (only for xlmr currently).",
    )
    return parser


def build_test_cli(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--file",
        help="The path of a csv dataset containing utterances and labels. If not provided, we look for files in data/classification/datasets.",
    )
    parser.add_argument("--lang", help="The language code to use for the dataset.")
    parser.add_argument(
        "--project", help="The project scope to which the dataset belongs."
    )
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Tune the classifier's threshold based on test results.",
    )
    return parser


def build_repl_cli(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--lang",
        help="Run the models and pre-processing for the given language code.",
        required=True,
    )
    return parser


def build_setup_prompt_cli(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--file",
        help="NLS-Key File downloaded from studio (.csv) or from flow-creator (.yaml)",
        required=True,
    )
    parser.add_argument(
        "--dest",
        help="Dest File (.yaml) to store the state-prompt-mapping",
        required=False,
    )
    parser.add_argument(
        "--config_path",
        help="Path to config.yaml (By default it is config/config.yaml)",
        required=False,
    )
    parser.add_argument(
        "--overwrite",
        help="If set to True, overwrites the prompts.yaml file in config",
        required=False,
    )
    return parser


def build_fill_nls_col_cli(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--input_file",
        help="Train or test file (.csv)",
        required=True,
    )
    parser.add_argument(
        "--output_file",
        help="Dest File (.csv) to save updates",
        required=False,
    )
    parser.add_argument(
        "--overwrite",
        help="If set to True, overwrites input_file",
        required=False,
    )
    return parser


def parse_commands(command_string: Optional[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    command_parsers = parser.add_subparsers(dest="command", help="Project utilities.")
    dir_cli_parser = command_parsers.add_parser(
        "setup-dirs", help="Create base directory structure."
    )
    data_split_cli_parser = command_parsers.add_parser(
        "split-data", help="Split a dataset into train-test datasets for given ratio."
    )
    data_combine_cli_parser = command_parsers.add_parser(
        "combine-data", help="Combine datasets into a single file."
    )
    train_cli_parser = command_parsers.add_parser("train", help="Train a workflow.")
    dev_cli_parser = command_parsers.add_parser("dev", help="Develop a workflow.")
    test_cli_parser = command_parsers.add_parser("test", help="Test a workflow.")
    repl_cli_parser = command_parsers.add_parser(
        "repl", help="Read Eval Print Loop for a trained workflow."
    )
    setup_prompt_cli_parser = command_parsers.add_parser(
        "setup-prompts", help="Make prompts.yaml mapping file from nls-keys"
    )
    fill_nls_col_cli_parser = command_parsers.add_parser(
        "get-nls-labels", help="Populate nls_label column in your train/test dataframes."
    )
    
    data_split_cli_parser = build_split_data_cli(data_split_cli_parser)
    data_combine_cli_parser = build_data_combine_cli(data_combine_cli_parser)
    train_cli_parser = build_train_cli(train_cli_parser)
    test_cli_parser = build_test_cli(test_cli_parser)
    dev_cli_parser = build_dev_cli(dev_cli_parser)
    repl_cli_parser = build_repl_cli(repl_cli_parser)
    setup_prompt_cli_parser = build_setup_prompt_cli(setup_prompt_cli_parser)
    fill_nls_col_cli_parser = build_fill_nls_col_cli(fill_nls_col_cli_parser)
    
    command = command_string.split() if command_string else None
    return parser.parse_args(command)


def main(command_string: Optional[str] = None) -> None:
    args = parse_commands(command_string=command_string)
    if args.command == "setup-dirs":
        create_data_directory(args)
    elif args.command == "split-data":
        create_data_splits(args)
    elif args.command == "combine-data":
        merge_datasets(args)
    elif args.command == "train":
        train_intent_classifier(args)
    elif args.command == "dev":
        dev_workflow(args)
    elif args.command == "test":
        test_classifier(args)
    elif args.command == "repl":
        repl(args)
    elif args.command == "setup-prompts":
        setup_prompts(args)
    elif args.command == "get-nls-labels":
        fill_nls_col(args)
    else:
        raise ValueError("Unrecognized command: {}".format(args.command))
