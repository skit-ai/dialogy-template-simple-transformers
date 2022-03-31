import os
import traceback
from typing import Any, Dict, List

from fastapi.responses import JSONResponse

import sentry_sdk
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware

from slu import constants as const
from slu.src.api import app, Input
from slu.src.controller.prediction import get_predictions
from slu.utils import error_response
from slu.utils.config import Config, YAMLLocalConfig
from slu.utils.sentry import capture_exception

CONFIG_MAP = YAMLLocalConfig().generate()
PREDICT_API = get_predictions(const.PRODUCTION)

if os.environ.get(const.ENVIRONMENT) == const.PRODUCTION:
    sentry_sdk.init(
        dsn=os.environ["SENTRY_DSN"],
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        # We recommend adjusting this value in production.
        # By default the SDK will try to use the SENTRY_RELEASE
        # environment variable, or infer a git commit
        # SHA as release, however you may want to set
        # something more human-readable.
        # release="myapp@1.0.0",
    )
    app.add_middleware(SentryAsgiMiddleware)


@app.get("/")
async def health_check():
    """
    Get server status health.

    The purpose of this API is to help other people/machines know liveness of the application.
    """

    return JSONResponse(
        dict(
            status="ok",
            response={"message": "Server is up."},
        ),
        status_code=200,
    )


@app.post("/predict/{lang}/{model_name}/")
async def slu(lang: str, model_name: str, payload: Input):
    """
    Get SLU predictions.
    Produces a json response containing intents and entities.
    """
    request = payload.dict()
    config: Config = list(CONFIG_MAP.values()).pop()

    if lang not in config.get_supported_languages():
        return error_response.invalid_language(lang)

    if not request[const.ALTERNATIVES] and not request[const.TEXT]:
        return error_response.invalid_input(request)

    try:
        utterance: Any = request[const.ALTERNATIVES] or request[const.TEXT]
        context: Dict[str, Any] = request.get(const.CONTEXT) or {}
        history: List[Dict[str, Any]] = request.get(const.HISTORY) or []
        intents_info: List[Dict[str, Any]] = request.get(const.INTENTS_INFO) or []

        try:
            response = PREDICT_API(
                alternatives=utterance,
                context=context,
                intents_info=intents_info,
                history=history,
                lang=lang,
            )
            history.append(response)
            return JSONResponse(
                dict(status="ok", response=response, history=history), status_code=200
            )

        except OSError as os_error:
            return error_response.missing_models(os_error)

    except Exception as exc:
        # Update this section to:
        # 1. Handle specific errors
        # 2. provide user-friendly messages. The current is developer friendly.
        # capture_exception(exc, ctx="api", message=request.json)
        capture_exception(exc, ctx="api", message=request)
        return JSONResponse(
            {"message": str(exc), "cause": traceback.format_exc()}, status_code=200
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0")
