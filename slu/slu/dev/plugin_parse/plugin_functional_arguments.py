import importlib
from typing import Any, List, Tuple, Union

fn_module = importlib.import_module("slu.dev.plugin_parse.plugin_functions")


def plugin_param_parser(value: List[Any]) -> Union[Any, Tuple[str, Tuple]]:
    """
    Plugins in the config can be described to have functional arguments
    using a certain format.

    :param value: A list expression.
    :type value: List[Any]
    """
    fn_triggers = ["access", "mutate", "env"]

    if not isinstance(value, list):
        return value

    if len(value) < 2 or len(value) > 3:
        return value

    if value[0] not in fn_triggers:
        return value

    fn_name = value[0]
    is_execute_now = value[-1] == []
    func = getattr(fn_module, fn_name)
    if is_execute_now:
        return func(*value[1:-1][0])

    node, arguments = value[1]
    return func(node, *arguments)
