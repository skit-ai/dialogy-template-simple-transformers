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
