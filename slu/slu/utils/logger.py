import sys

from loguru import logger



sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

config = {
    "handlers": [
        {
            "sink": sys.stdout,
            "format": """
-------------------------------------------------------
<level>{level}</level>
-------
TIME: <green>{time}</green>
FILE: {name}:L{line} <blue>{function}(...)</blue>
<level>{message}</level>
-------------------------------------------------------
""",
            "colorize": True,
        },
        {
            "sink": "file.log",
            "rotation": "500MB",
            "retention": "10 days",
            "encoding": "utf8",
            "format": """
-------------------------------------------------------
<level>{level}</level>
-------
TIME: <green>{time}</green>
FILE: {name}:L{line} <blue>{function}(...)</blue>
<level>{message}</level>
-------------------------------------------------------
""",
        },
    ]
}


logger.configure(**config)
logger.enable("slu")
