import os
from typing import Any

from dialogy.workflow import Workflow

from slu import constants as const


def access(node: str, *attributes: str):
    def read(workflow):
        workflow_io = getattr(workflow, node)
        return (workflow_io[attribute] for attribute in attributes)

    return read


def mutate(node: str, attribute: str, action=const.EXTEND):
    def write(workflow: Workflow, value: Any):
        workflow_io = getattr(workflow, node)
        if action == const.REPLACE:
            workflow_io[attribute] = value
        elif action == const.EXTEND:
            workflow_io[attribute].extend(value)

    return write


def env(*attributes: str) -> Any:
    attribute = attributes[0]
    return os.environ.get(attribute)


def duckling_access(w: Workflow):
    current_state = w.input[const.CONTEXT].get(const.CURRENT_STATE)
    use_latent = current_state in []
    return w.input[const.NER_INPUT], w.input[const.REFERENCE_TIME], w.input[const.LOCALE], use_latent
