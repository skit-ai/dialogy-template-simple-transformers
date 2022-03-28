from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel


class Input(BaseModel):
    alternatives: Optional[
        Union[List[List[Dict[str, Any]]], List[Dict[str, Any]]]
    ] = None
    context: Optional[Dict[str, Any]] = {}
    history: Optional[List[Dict[str, Any]]] = []
    intents_info: Optional[List[Dict[str, Any]]] = []
    short_utterance: Optional[Dict[Any, Any]] = None
    text: Optional[str] = None
