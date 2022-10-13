import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import psutil
import pytz
from slack_sdk import WebClient

from slu import constants as const


def send_slack_notif(
    client: WebClient,
    channel: str,
    default_notif: str,
    max_notifs: int = const.MAX_NOTIFS,
    notification_cooldown: int = const.NOTIFICATION_COOLDOWN,
    error_notif: Optional[str] = None,
    blocks: List[Dict[str, Any]] = None,
):
    current_time = datetime.now(pytz.timezone("UTC"))
    current_ts = current_time.timestamp()
    svc_start_ts = psutil.Process(os.getpid()).create_time()
    svc_recently_started = current_ts - svc_start_ts < 10
    oldest_message_ts = (
        current_time - timedelta(minutes=notification_cooldown)
    ).timestamp()
    conv_history = client.conversations_history(
        limit=100, channel=channel, oldest=oldest_message_ts, inclusive=True
    )

    error_notif = error_notif or default_notif
    msgs = conv_history.get("messages")
    n_repeats = 0

    # First message is latest
    for msg in msgs:
        msg_text = msg.get("text")
        bot_id = msg.get("bot_id")

        # Search for the most recent message from the bot ...
        if not bot_id:
            continue

        # ... same as the current message
        if msg_text == default_notif or msg_text == error_notif:
            n_repeats += 1

        # but if we have sent it `max_notifs` times, then we don't search further
        if n_repeats > max_notifs:
            break

    text = error_notif if n_repeats and svc_recently_started else default_notif
    payload = {"channel": channel, "text": text, "blocks": blocks}

    if n_repeats <= max_notifs:
        client.chat_postMessage(**payload)
