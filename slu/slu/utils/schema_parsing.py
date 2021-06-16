from typing import Any, Dict, List
import pandas as pd
from slu import constants as const
from slu.utils.config import Config


def list_entity_plugin_schema_parser(entity_type: str, config: Config, entity_config_file: Any) -> Dict[str, Any]:
    """
    Parse csv file to obtain list_entity_plugin_schema.

    :param entity_config_file: A csv file containing patterns and values.
    :type entity_config_file: Any
    """
    pattern_df: pd.DataFrame = pd.read_csv(entity_config_file)

    # We are expecting two columns.
    columns = pattern_df.columns
    pattern_column = columns[0]
    value_column = columns[1]
    candidates: Dict[str, Dict[str, List[str]]] = {entity_type: {}}

    for _, row in pattern_df.iterrows():
        entity_value = row[value_column]
        pattern = row[pattern_column]
        if entity_value in candidates[entity_type]:
            candidates[entity_type][entity_value].append(pattern)
        else:
            candidates[entity_type][entity_value] = [pattern]

    config.update_plugin_metadata(const.LIST_ENTITY_PLUGIN, const.CANDIDATES, candidates)
    return config.json()


plugin_schema_parser_map = {
    const.LIST_ENTITY_PLUGIN: list_entity_plugin_schema_parser
}


def schema_parser(plugin: str, entity_type: str, config: Config, entity_config_file: Any) -> Dict[str, Any]:
    """
    Parse entity config for each plugin.

    :param plugin: A supported plugin.
    :type plugin: str
    :param entity_type: [description]
    :type entity_type: str
    :param config: [description]
    :type config: Config
    :param entity_config_file: A file object containing data-structure that can be parsed.
    :type entity_config_file: Any
    :return: Updated slu-service config.
    :rtype: Any
    """
    return plugin_schema_parser_map[plugin](entity_type, config, entity_config_file)
