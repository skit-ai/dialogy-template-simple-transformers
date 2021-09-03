import sys

from loguru import logger

import slu.constants as const

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
