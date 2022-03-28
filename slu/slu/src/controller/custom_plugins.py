from typing import Any, Optional, List, Dict

from dialogy.base.plugin import Plugin, Guard, Input, Output
from dialogy.types import Intent, BaseEntity

import slu.constants as const


class OOSFilterPlugin(Plugin):
    def __init__(self, dest=None, guards=None, threshold=None, **kwargs) -> None:
        super().__init__(dest=dest, guards=guards, **kwargs)
        self.threshold = threshold

    def set_oos_intent(self, intents: List[Intent]) -> Any:
        if intents[0].score < self.threshold:
            intents[0].name = const.INTENT_OOS
        return intents

    def utility(self, input_: Input, output: Output) -> Any:
        return self.set_oos_intent(output.intents)


class ContextualIntentSwap(Plugin):
    def __init__(self, dest=None, **kwargs) -> None:
        super().__init__(dest=dest, **kwargs)

    def swap(
        self, intents: List[Intent], context: Dict[str, Any], entities: List[BaseEntity]
    ) -> Any:
        """
        Swap the predicted intent with another basis the context.

        This is a temporary solution to the problem of the context management, we want this to be solved via Dialog management.
        The utility is to evaluate some conditions and rename an intent. An example:

        Case I:
        BOT: What is your favorite color?
        USER: Red

        The intent in the above case would be `_inform_`.

        Case II:
        BOT: What is your favorite color?
        USER: My favourite colour is blue.

        The intent in the above case would be `colour_preference`.

        To produce the same response for both the conversations, we should ideally let the dialog manager expect both intents in the same state.
        ** This is not possible in older projects on legacy systems. This plugin is meant to support only those cases. **

        We rely on `current_intent`, `state` and `entity` type and/or values to decode the new intent name.
        """
        if not isinstance(intents, list):
            raise TypeError(f"Intents must be a list, not {type(intents)}")
        if not intents:
            raise ValueError("No intents provided")
        if not isinstance(intents[0], Intent):
            raise TypeError(
                f"Each Intent must be of type Intent within {intents}, not {type(intents[0])}"
            )

        intent = intents[0]
        tracked_intent = context[const.CURRENT_INTENT]

        if tracked_intent == "" and intent.name == "":
            intent.name = tracked_intent

        return intents

    def utility(self, *args: Any) -> Any:
        return super().utility(*args)
