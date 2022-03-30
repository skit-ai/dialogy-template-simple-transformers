import os
import traceback
from typing import Any, Dict, List

import sentry_sdk
from flask import jsonify, request
from sentry_sdk.integrations.flask import FlaskIntegration

from slu import constants as const
from slu.src.api import app
from slu.src.api.thread_safe import ThreadSafePredictAPI
from slu.utils import error_response
from slu.utils.config import Config, YAMLLocalConfig
from slu.utils.sentry import capture_exception
from slu.src.controller.prediction import get_predictions


CONFIG_MAP = YAMLLocalConfig().generate()
CONFIG: Config = list(CONFIG_MAP.values()).pop()
PREDICT_API = get_predictions(const.PRODUCTION, config=CONFIG)


if os.environ.get(const.ENVIRONMENT) == const.PRODUCTION:
    sentry_sdk.init(
        dsn=os.environ["SENTRY_DSN"],
        integrations=[FlaskIntegration()],
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        # We recommend adjusting this value in production.
        # By default the SDK will try to use the SENTRY_RELEASE
        # environment variable, or infer a git commit
        # SHA as release, however you may want to set
        # something more human-readable.
        # release="myapp@1.0.0",
    )


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


@app.route("/predict/<lang>/<model_name>/", methods=["POST"])
def slu(lang: str, model_name: str):
    """
    Get SLU predictions.

    Produces a json response containing intents and entities.
    """
    config: Config = list(CONFIG_MAP.values()).pop()
    if lang not in config.get_supported_languages():
        return error_response.invalid_language(lang)

    if not isinstance(request.json, dict):
        return error_response.invalid_request(request.json)

    if not (const.ALTERNATIVES in request.json or const.TEXT in request.json):
        return error_response.invalid_input(request.json)

    try:
        utterance: Any = request.json.get(const.ALTERNATIVES) or request.json.get(
            const.TEXT
        )

        context: str = request.json.get(const.CONTEXT) or {}  # type: ignore
        intents_info: List[Dict[str, Any]] = request.json.get(const.INTENTS_INFO) or []
        history: List[Any] = request.json.get(const.HISTORY) or []

        with ThreadSafePredictAPI(
            utterance,
            lang,
            PREDICT_API,
            context=context,
            intents_info=intents_info,
            history=history
        ) as response:
            history.append(response)
            return jsonify(status="ok", response=response, history=history), 200

    except Exception as exc:
        # Update this section to:
        # 1. Handle specific errors
        # 2. provide user-friendly messages. The current is developer friendly.
        capture_exception(exc, ctx="api", message=request.json)
        return jsonify({"message": str(exc), "cause": traceback.format_exc()}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0")
