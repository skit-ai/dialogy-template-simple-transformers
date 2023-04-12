import os
import json
import datetime

from typing import List

from pydantic import BaseModel
from requests import post
from requests.exceptions import ConnectionError

from dialogy.base import Input, Output, Plugin

from slu.utils import logger


class Result(BaseModel):
    input: Input
    output: Output


class PluginProxy(Plugin):
    def __init__(self, plugin_name, **kwargs) -> None:
        self.plugin_name = plugin_name
        super().__init__(**kwargs)

    def __call__(self, input: Input, output: Output):
        return slu_customization_single(self.plugin_name, input, output)

    def utility(self, input_: Input, output: Output):
        pass


class PluginProxyFused(Plugin):
    def __init__(self, plugins: List[str], **kwargs) -> None:
        self.plugins = plugins
        super().__init__(**kwargs)

    def __call__(self, input: Input, output: Output):
        return slu_customization_fused(self.plugins, input, output)

    def utility(self, input_: Input, output: Output):
        pass


def default(o):
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()


def slu_customization_single(plugin_name, input, output):
    try:
        response = post(
            os.environ.get("CUSTOMIZATION_URL", "http://localhost:9999")
            + "/single/<slu-name>/"
            + plugin_name,
            data=json.dumps(
                {
                    "input": input.dict(),
                    "output": output.dict()
                },
                default=default
            )
        )

        if response.status_code == 200:
            result = Result(**response.json())
            return result.input, result.output
        else:
            raise ValueError(response.json())

    except ConnectionError as connection_error:
        msg = "SLU Customization server is unreachable"
        logger.critical(msg + ": " + str(connection_error))
        raise ConnectionError(msg, connection_error)


def slu_customization_fused(plugins, input, output):
    try:
        response = post(
            os.environ.get("CUSTOMIZATION_URL", "http://localhost:9999")
            + "/fused/<slu-name>",
            data=json.dumps(
                {
                    "plugins": plugins,
                    "input": input.dict(),
                    "output": output.dict()
                },
                default=default
            )
        )

        if response.status_code == 200:
            result = Result(**response.json())
            return result.input, result.output
        else:
            raise ValueError(response.json())

    except ConnectionError as connection_error:
        msg = "SLU Customization server is unreachable"
        logger.critical(msg + ": " + str(connection_error))
        raise ConnectionError(msg, connection_error)
