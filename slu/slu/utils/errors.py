from flask import jsonify, request

from slu import constants as const


def invalid_language(supported_languages):
    return (
            jsonify(
                {
                    "message": "Invalid request.",
                    "cause": f"Language is not supported. Support available only for {supported_languages}.",
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
