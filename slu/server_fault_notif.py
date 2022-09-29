"""
Send an alert message to a slack channel.
"""
import os
import pytz
import psutil
from datetime import datetime, timedelta

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


CHANNEL = os.getenv("CHANNEL")
POD = os.getenv("HOSTNAME")
SLACK_TOKEN = os.getenv("SLACK_TOKEN")
AUTHOR = os.getenv("AUTHOR")
NAME_PARTS = POD.split("-") if POD else ""
SVC_NAME = "-".join(NAME_PARTS[:-2])

try:
    error_msg = f"<@{AUTHOR}> {SVC_NAME} crashed! :feelsstrongman:"
    client = WebClient(token=SLACK_TOKEN)
    current_time = datetime.now(pytz.timezone("UTC"))
    current_ts = current_time.timestamp()
    svc_start_ts = psutil.Process(os.getpid()).create_time()
    svc_recently_started = current_ts - svc_start_ts < 10
    oldest_message_ts = (current_time - timedelta(minutes=20)).timestamp()
    conv_history = client.conversations_history(
        limit=100, channel=CHANNEL, oldest=oldest_message_ts, inclusive=True
    )
    msgs = conv_history.get("messages")

    n_repeats = 0
    for msg in msgs:
        msg_text = msg.get("text")
        bot_id = msg.get("bot_id")

        # Search for the most recent message from the bot ...
        if not bot_id:
            continue

        # ... same as the current message
        if msg_text == error_msg:
            n_repeats += 1

        # but if we have sent it 2 times, then we don't search further
        if n_repeats > 2:
            break

    client.chat_postMessage(
        channel=CHANNEL,
        text=error_msg,
    )
except SlackApiError as error:
    pass
