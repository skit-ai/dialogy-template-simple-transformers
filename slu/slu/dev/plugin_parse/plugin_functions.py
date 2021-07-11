import os
from typing import Any
from dialogy.workflow import Workflow


def access(node: str, *attributes: str):
    def read(workflow):
        workflow_io = getattr(workflow, node)
        return (workflow_io[attribute] for attribute in attributes)
    return read


def mutate(node: str, *attributes: str):
    attribute = attributes[0]
    def write(workflow: Workflow, value: Any):
        workflow_io = getattr(workflow, node)
        container = workflow_io[attribute]
        if isinstance(container, list):
            if isinstance(value, list):
                container += value
            else:
                container.append(value)
        else:
            workflow_io[attribute] = value
    return write


def env(*attributes: str) -> Any:
    attribute = attributes[0]
    return os.environ.get(attribute)
