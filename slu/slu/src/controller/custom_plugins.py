from typing import Any, Optional, List, Dict
from datetime import datetime

from dialogy.base.plugin import Plugin, PluginFn
from dialogy.types import Intent, BaseEntity, TimeEntity
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


class CombineDateWithTime(Plugin):
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"

    def __init__(
        self,
        access: Optional[PluginFn] = None,
        mutate: Optional[PluginFn] = None,
        input_column: str = const.ALTERNATIVES,
        output_column: Optional[str] = None,
        use_transform: bool = False,
        trigger_intents: Optional[List[str]] = None,
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
        self.trigger_intents = trigger_intents
        self.trigger_entity_types = [CombineDateWithTime.DATE, CombineDateWithTime.TIME]

    def join(self, entity: TimeEntity, previous_entity: TimeEntity):
        current_datetime = entity.get_value()
        previous_datetime = entity.get_value()
        entity.type = CombineDateWithTime.DATETIME

        if entity.type == CombineDateWithTime.DATE:
            combined_value = current_datetime.replace(hour=previous_datetime.hour, minute=previous_datetime.minute, second=previous_datetime.second)
        elif entity.type == CombineDateWithTime.TIME:
            combined_value = current_datetime.replace(year=previous_datetime.year, month=previous_datetime.month, day=previous_datetime.day)

        entity.value = combined_value.isoformat()

    def utility(self, intents_info: List[Dict[str, Any]], entities: List[BaseEntity]) -> Any:
        """
        Combine the date and time entities into a single entity.

        This is a temporary solution to the problem of the context management, we want this to be solved via Dialog management.
        The utility is to evaluate some conditions and rename an intent. An example:

        Turn 0:
        BOT: When do you want to visit?
        USER: Tomorrow

        Turn 1:
        BOT: and what time?
        USER: at 4pm
        """
        previous_entity = None
        previous_intent = None

        if not self.trigger_intents:
            return

        if not self.trigger_entity_types:
            return

        for entity in entities:
            if entity.type in self.trigger_entity_types:
                previous_intents = [intent for intent in intents_info if intent[const.NAME] in self.trigger_intents]

                if not previous_intents:
                    continue

                previous_intent = previous_intents[0]
                entities = previous_intent[const.SLOTS]

                if not entities:
                    continue

                previous_entity = entities[0]
                self.join(entity, previous_entity)
                break
