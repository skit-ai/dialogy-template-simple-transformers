import sentry_sdk


def capture_exception(e: Exception, ctx="global", message="NOT-SET"):
    """
    Capture the exception on Sentry explicitly.

    Args:
        e (Exception): Any instance of super-type Exception.
        ctx (str, optional): Context - so that logs on sentry can be more helpfule. Defaults to "global".
        message (str, optional): helps finding extra details other than the exceptions' name. Defaults to "NOT-SET".
    """
    sentry_sdk.capture_exception(e)
    sentry_sdk.set_context(
        ctx,
        {
            "name": "slu",
            "message": message,
        },
    )
