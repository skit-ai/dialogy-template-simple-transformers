import traceback
from datetime import datetime
from typing import Any, Dict, List

from dialogy.preprocess.text.normalize_utterance import normalize
from flask import jsonify, request

from slu import constants as const
from slu.src.api import app
from slu.src.controller.prediction import predict_wrapper
from slu.utils.config import HTTPConfig, YAMLLocalConfig, Config
from slu.utils.sentry import capture_exception
from slu.utils import error_response
from slu.utils.schema_parsing import schema_parser


CLIENT_CONFIGS = YAMLLocalConfig().generate()
PREDICT_API = predict_wrapper(CLIENT_CONFIGS)


@app.route("/", methods=["GET"])
def health_check():
    """
    Get server status health.

    The purpose of this API is to help other people/machines know liveness of the application.
    """
    return jsonify(
        status="ok",
        response={"message": "Server is up."},
    )


@app.route("/predict/<lang>/<client_name>/<model_name>/", methods=["POST"])
def slu(lang: str, client_name: str, model_name: str):
    """
    Get SLU predictions.

    Produces a json response containing intents and entities.
    """
    config = CLIENT_CONFIGS.get(model_name, None)

    if config is None:
        return error_response.missing_project_name(model_name), 404

    if lang not in config.get_supported_languages():
        return error_response.invalid_language(lang)

    if not isinstance(request.json, dict):
        return error_response.invalid_request(request.json)

    if not (const.ALTERNATIVES in request.json or const.TEXT in request.json):
        return error_response.invalid_input(request.json)

    try:
        maybe_utterance: Any = request.json.get(const.ALTERNATIVES) or request.json.get(
            const.TEXT
        )

        sentences: List[str] = normalize(maybe_utterance)
        context: str = request.json.get(const.CONTEXT) or {} # type: ignore
        intents_info: List[Dict[str, Any]] = (
            request.json.get(const.S_INTENTS_INFO) or []
        )

        response = PREDICT_API(
            config,
            sentences,
            context,
            intents_info=intents_info,
            reference_time=int(datetime.now().timestamp() * 1000),
            locale=const.LANG_TO_LOCALES[lang]
        )

        return jsonify(status="ok", response=response), 200

    except Exception as exc:
        # Update this section to:
        # 1. Handle specific errors
        # 2. provide user-friendly messages. The current is developer friendly.
        capture_exception(exc, ctx="api", message=request.json)
        return jsonify({"message": str(exc), "cause": traceback.format_exc()}), 500


@app.route("/update/<client_name>/<model_name>/", methods=["POST"])
def update_config(client_name: str, model_name: str):
    """
    Update SLU config.
    """

    if not isinstance(request.json, dict):
        return error_response.invalid_request(request.json)

    CLIENT_CONFIGS[model_name] = Config(**request.json)

    return jsonify(
        status="ok",
        response={"message": f"{model_name} has been updated | ref {client_name}."},
    )


@app.route("/schema/<plugin>/<entity_type>/<client_name>/<model_name>/", methods=["POST"])
def create_plugin_schema(plugin: str, entity_type: str, client_name: str, model_name: str):
    """
    Generate plugin config from a readable source.

    :param plugin: Dialogy plugin or lambda function.
    :type plugin: str
    :param entity_type: The entity for which we need to produce plugin schema.
    :type entity_type: str
    :param client_name: Required for readability and debugging.
    :type client_name: str
    :param model_name: The project for which we need to save the config in the plugin.
    :type model_name: str
    """
    if model_name not in CLIENT_CONFIGS:
        return error_response.invalid_initialization(client_name, model_name)

    if plugin != const.LIST_ENTITY_PLUGIN:
        return error_response.unknown_plugin(plugin)

    entity_config_file = request.files.get("entity_config")

    if entity_config_file is None:
        return error_response.config_upload_required(plugin) 

    config = CLIENT_CONFIGS[model_name]

    updated_config = schema_parser(plugin, entity_type, config, entity_config_file)

    return jsonify(
        status="ok",
        response=updated_config,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0")
