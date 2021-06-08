from flask import jsonify, request

from slu import constants as const


def invalid_language(lang):
    return (
            jsonify(
                {
                    "message": "Invalid request.",
                    "cause": f"Language '{lang}' is not supported.",
                }
            ),
            400,
        )


def invalid_request(req):
    return (
        jsonify(
            {
                "message": "Invalid request.",
                "cause": f"Post body should be a dictionary, received {type(req)}.",
            }
        ),
        400,
    )


def invalid_input(req):
    return (
        jsonify(
            {
                "message": "Invalid request.",
                "cause": f"Post body should have either of these keys: {const.TEXT}, {const.ALTERNATIVES}."
                         f" Instead got\n{req}",
            }
        ),
        400,
    )

def missing_project_name(project_name):
    return (
        jsonify(
            {
                "message": "Project not found.",
                "cause": f"config for project_name: {project_name} is not present."
            }
        ),
        404,
    )
