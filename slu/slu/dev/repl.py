"""
This module offers an interactive repl to run a Workflow.
"""
import re
import argparse
import json
from pprint import pprint

from dialogy.utils import normalize
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory

import slu.constants as const
from slu.dev.version import check_version_save_config
from slu.src.controller.prediction import get_predictions
from slu.utils import logger
from slu.utils.config import Config, YAMLLocalConfig


def repl_prompt(separator="", show_help=True):
    message = """
    Provide either a json like:\n

    ```
    {
        "alternatives": [[{"transcript": "...", "confidence": "..."}]],
        "context": {}
    }
    ```

    or

    ```
    [[{"transcript": "...", "confidence": "..."}]]
    ```

    or just plain-text: "This sentence gets converted to above internally!"

Input interactions:

- ESC-ENTER to submit
- C-c or C-d to exit (C = Ctrl)
    """
    message = message.strip()
    return f"{message}\n{separator}\nEnter>\n" if show_help else "Enter>\n"


def make_input(input_string):
    # Check if this is a json compatible input
    # if yes, does it have alternatives and context?
    # or is it an Utterance which can be normalized?
    try:
        payload = json.loads(input_string)
        if const.ALTERNATIVES in payload:
            return payload
        else:
            return {const.ALTERNATIVES: payload}
    except json.JSONDecodeError:
        input_string = re.sub(r"(\s+|\n+)", " ", input_string).strip()
        return {const.ALTERNATIVES: normalize(input_string)}


def repl(args: argparse.Namespace) -> None:
    version = args.version
    lang = args.lang
    project_config_map = YAMLLocalConfig().generate()
    config: Config = list(project_config_map.values()).pop()
    check_version_save_config(config, version)

    separator = "-" * 100
    show_help = True
    logger.info("Loading models... this takes around 20s.")
    session = PromptSession(history=FileHistory(".repl_history"))  # type: ignore
    prompt = session.prompt
    auto_suggest = AutoSuggestFromHistory()
    PREDICT_API = get_predictions(const.PRODUCTION, config=config)
    show_help = True

    try:
        while True:
            raw = prompt(
                repl_prompt(separator=separator, show_help=show_help),
                multiline=True,
                auto_suggest=auto_suggest,
            )

            if raw == "--help":
                raw = prompt(
                    repl_prompt(separator=separator, show_help=True),
                    multiline=True,
                    auto_suggest=auto_suggest,
                )
            input_ = make_input(raw)

            response = PREDICT_API(
                **input_,
                lang=lang,
            )
            pprint(response)
            show_help = False
    except KeyboardInterrupt:
        logger.info("Exiting...")
    except EOFError:
        logger.info("Exiting...")
