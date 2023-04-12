import json
import os
import traceback
from typing import Any, Dict, List

import sentry_sdk
from fastapi import BackgroundTasks
from fastapi.responses import JSONResponse, Response
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from slack_sdk import WebClient

from slu import constants as const
from slu.src.api import Input, app
from slu.src.controller.prediction import get_predictions
from slu.utils import error_response
from slu.utils.config import Config, YAMLLocalConfig
from slu.utils.sentry import capture_exception
from slu.utils.slack import send_slack_notif

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


@app.get("/health/{probe_type}")
async def health_check(probe_type, background_tasks: BackgroundTasks):
    """
    Get server status health.

    The purpose of this API is to help other people/machines know liveness of the application.
    """
    try:
        channel = os.getenv("CHANNEL")
        pod = os.getenv("HOSTNAME")
        name_parts = pod.split("-")
        svc_name = "-".join(name_parts[:-2])
        author = os.getenv("AUTHOR")
        slack_client = WebClient(token=os.getenv("SLACK_TOKEN"))
        startup_msg = f"<@{author}> {svc_name} is running :white_check_mark:"
        container_restart_msg = f"<@{author}> {svc_name} restarted :sadpepe:"
        container_exp_msg = (
            f"<@{author}> {svc_name} {probe_type} check failed :feelsstrongman:"
        )

        with open(f"config/request.{probe_type}.json", "r") as f:
            data = json.load(f)

        payload = data["payload"]
        lang = data["lang"]
        expected_intent = data.get("expected_intent")
        expected_entity_type = data.get("expected_entity_type")
        expected_entity_value = data.get("expected_entity_value")

        response = PREDICT_API(
            **payload,
            lang=lang,
        )

        intent = response[const.INTENTS][0]

        assert (
                intent["name"] == expected_intent
        ), f"{expected_intent=},{intent['name']=} "

        if expected_entity_type and expected_entity_value:
            entity_type = intent["slots"][0]["values"][0]["entity_type"]
            entity_value = intent["slots"][0]["values"][0]["value"]

            assert (
                    entity_type == expected_entity_type
            ), f"{expected_entity_type=},{entity_type=} "
            assert (
                    entity_value == expected_entity_value
            ), f"{expected_entity_value=},{entity_value=} "

        if probe_type == "startup":
            background_tasks.add_task(
                send_slack_notif,
                slack_client,
                channel,
                startup_msg,
                error_notif=container_restart_msg,
            )

        return Response(status_code=200)

    except Exception as e:
        background_tasks.add_task(
            send_slack_notif,
            slack_client,
            channel,
            container_exp_msg,
            blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"""
<@{author}> {pod} {probe_type} check failed!  

```
{traceback.format_exc()}
```

                        """.strip(),
                    },
                }
            ],
        )

        return JSONResponse(
            dict(status="ok", message=str(e), cause=traceback.format_exc()),
            status_code=500,
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

    except FileNotFoundError as io_error:
        return error_response.missing_models(io_error)

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
