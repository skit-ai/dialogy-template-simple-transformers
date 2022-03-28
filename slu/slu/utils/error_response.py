from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from slu import constants as const


def invalid_language(lang):
    return JSONResponse (
        content=jsonable_encoder(
            {
                "message": "Invalid request.",
                "cause": f"Language '{lang}' is not supported.",
            }
        ),
        status_code=400,
    )


def invalid_request(req):
    return JSONResponse (
        content=jsonable_encoder(
            {
                "message": "Invalid request.",
                "cause": f"Post body should be a dictionary, received {type(req)}.",
            }
        ),
        status_code=400,
    )


def invalid_input(req):
    return JSONResponse (
        content=jsonable_encoder(
            {
                "message": "Invalid request.",
                "cause": f"Post body should have either of these keys: {const.TEXT}, {const.ALTERNATIVES}."
                f" Instead got\n{req}",
            }
        ),
        status_code=400,
    )


def missing_project_name(project_name):
    return JSONResponse (
        content=jsonable_encoder(
            {
                "message": "Project not found.",
                "cause": f"config for project_name: {project_name} is not present.",
            }
        ),
        status_code=400,
    )


def invalid_initialization(client_name, model_name):
    return JSONResponse (
        content=jsonable_encoder(
            {
                "message": "Missing config.",
                "cause": f"Server started but config for {client_name} -- {model_name} was not loaded.",
            }
        ),
        status_code=500,
    )


def config_upload_required(plugin):
    return JSONResponse (
        content=jsonable_encoder(
            {
                "message": "Plugin needs config file.",
                "cause": f"Plugin {plugin} needs a config file.",
            }
        ),
        status_code=400,
    )


def unknown_plugin(plugin):
    return JSONResponse (
        content=jsonable_encoder(
            {
                "message": "No schema available for plugin.",
                "cause": f"Plugin {plugin} does not have a schema parser defined.",
            }
        ),
        status_code=400,
    )


def missing_models(message):
    return JSONResponse (
        content=jsonable_encoder(
            {
                "message": message,
                "cause": "Possibly missed the training step or it wasn't planned.",
            }
        ),
        status_code=500,
    )
