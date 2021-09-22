from typing import Any, Optional, List, Dict

from dialogy.base.plugin import Plugin, PluginFn
from dialogy.types import Intent, BaseEntity
import slu.constants as const


class ContextualIntentSwap(Plugin):
    def __init__(
        self,
        access: Optional[PluginFn] = None,
        mutate: Optional[PluginFn] = None,
        input_column: str = const.ALTERNATIVES,
        output_column: Optional[str] = None,
        use_transform: bool = False,
        debug: bool = False
    ) -> None:
        super().__init__(
            access=access,
            mutate=mutate,
            input_column=input_column,
            output_column=output_column,
            use_transform=use_transform,
            debug=debug
        )

    def swap(self, intents: List[Intent], context: Dict[str, Any], entities: List[BaseEntity]) -> Any:
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
            raise TypeError(f"Each Intent must be of type Intent within {intents}, not {type(intents[0])}")

        intent = intents[0]
        tracked_intent = context[const.CURRENT_INTENT]

        if tracked_intent == "" and intent.name == "":
            intent.name = tracked_intent

        return intents

    def utility(self, *args: Any) -> Any:
        return super().utility(*args)
