"""
Imports:

- XLMRWorkflow
"""
from typing import Any, Dict, List

import numpy as np
from dialogy.workflow import Workflow  # type: ignore
from dialogy.types.intent import Intent

from dialogy.types.entity import BaseEntity

from slu import constants as const
from slu.utils.logger import log
from slu.utils.config import Config
from slu.utils.sentry import capture_exception


class XLMRWorkflow(Workflow):
    """
    An instance of this class provides the workflow to access:

    - XLMR classifier
    - XLMR NER
    - Duckling entities

    Do note, XLMR NER would need parsing unlike Duckling which does detection and parsing both.
    To use Duckling, provide DucklingParser that ships with Dialogy as a preprocessor.
    """

    def __init__(
            self,
            preprocessors: Any,
            postprocessors: Any,
            debug: bool = False,
    ):
        """
        A workflow instance allows inference of an intent and a set of pre-configured entities.
        Slot filling applies as per rules.
        """
        super().__init__(
            preprocessors=preprocessors, postprocessors=postprocessors, debug=debug
        )

        # Read config/config.yaml and setup slu-level utils.
        self.config = Config()

        # XLMR Classifier
        self.classifier = self.config.get_model(const.CLASSIFICATION, const.PROD)

        # XLMR NER
        self.ner = self.config.get_model(const.NER, const.PROD)

        # You should extend dialogy.types.entity.BaseEntity
        # and use it to different types of entities here. Like:
        # {
        #   "location": LocationEntity, # (this ships with dialogy already!)
        #   "org_name": OrganizationEntity # (Extend BaseEntity to define this)
        # }
        self.entity_type_map: Dict[str, BaseEntity] = {
            "entity": None,
        }

        # Processed labels for classification tasks.
        self.labelencoder = self.config.load_pickle(
            const.CLASSIFICATION, const.S_INTENT_LABEL_ENCODER
        )

    def classify(self, text: str, fallback_intent=const.S_INTENT_ERROR) -> Intent:
        """
        Sentence to Intent classification.

        Returns:
            Intent: name and score (confidence) are the prominent attributes.
        """
        if not self.config.use_classifier:
            return Intent(name=const.S_INTENT_ERROR, score=1)

        predictions, raw_outputs = self.classifier.predict([text])

        try:
            # We will expect only one sentence within `texts`.
            predicted_intent = self.labelencoder.inverse_transform(predictions)[0]
            raw_output = raw_outputs[0]

            # Confidence estimate.
            confidence = max(np.exp(raw_output) / sum(np.exp(raw_output)))

            # Threshold's should also consider data samples available per class.
            # Using http://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/
            # should shed more light on optimal threshold usage.
            if confidence < self.config.classification_threshold:
                predicted_intent = fallback_intent
        except IndexError as index_error:
            # This exception means raw_outputs classifier failed to produce raw_outputs.
            predicted_intent = fallback_intent
            confidence = 1
            capture_exception(index_error, ctx="workflow", message="raw_outputs")

        return Intent(name=predicted_intent, score=confidence)

    def make_entity(
            self, entity_type: str,
            entity_values: List[Any],
            entity_starts: List[int],
            entity_ends: List[int],
            index: int,
            text: str
    ) -> BaseEntity:
        """
        Build entity from raw BIO labels and text tokens.

        Args:
            entity_type (str): The type of entity class that should be created.
            entity_values (Any): The value to be set for the entity.
            entity_starts (List[int]): The ranges that mark starting point of the entity token within the text.
            entity_ends (List[int]): The ranges that mark ending point of the entity token within the text
            index (int): The ASR alternative that references the text from which the entity will be made.
                            This helps in tracking the source of an entity.
            text (str): The string that is the source of BIO labels and entity tokens.

        Returns:
            BaseEntity: An entity object.
        """
        try:
            entity_class: BaseEntity = self.entity_type_map[entity_type]
            value = "_".join(entity_values)
            start = min(entity_starts)
            end = max(entity_ends)

            entity = entity_class(  # type: ignore
                type=entity_type,
                dim=entity_type,
                range={"start": start, "end": end},
                body=text[start:end],
                score=0,
                alternative_index=index,
                latent=False,
            )

            entity.set_value(value)  # type: ignore
            return entity
        except KeyError as key_error:
            raise KeyError("You need to configure entity "
                           f"classes for entity {entity_type}. "
                           "Refer to https://github.com/Vernacular-ai/dialogy/blob/master/dialogy/types/entity/base_entity.py")

    def combine_entity_groups(
            self, entity_groups: Dict[str, BaseEntity], index: int, text: str
    ) -> List[BaseEntity]:
        """
        Combine entities originating from compatible raw BIO entities.

        Args:
            entity_groups (Dict[str, BaseEntity]): Raw BIO entities grouped by type.
            index (int): A reference to the list index to which entity_groups belongs.
            text (str): A reference to the string in a list to which entity_groups belongs.

        Returns:
            List[BaseEntity]
        """
        entities: List[BaseEntity] = []

        for entity_type, entity_meta in entity_groups.items():
            entity_values: List[str] = []
            entity_starts: List[int] = []
            entity_ends: List[int] = []

            for i, (raw_type, (key, start, end)) in enumerate(entity_meta):
                # We are checking for B-<entity> this will terminate the compilation of
                # a previous entity. if a B-<entity-a> is recorded initially, we need to
                # look for its I-<entity-a> counterpart, and we log the start, end indices
                # during the process.
                #
                # It may happen that we find another B-<entity-b> before finding I-<entity-a>
                # in such cases we need to collect this B-<entity-b> separately instead of mixing
                # with <entity-b> content.
                if i > 0 and raw_type[0].upper() == "B":
                    entity = self.make_entity(
                        entity_type,
                        entity_values,
                        entity_starts,
                        entity_ends,
                        index,
                        text,
                    )
                    entity.add_parser(self.__class__)
                    entities.append(entity)

                entity_values.append(key)
                entity_starts.append(start)
                entity_ends.append(end)

            entity = self.make_entity(
                entity_type, entity_values, entity_starts, entity_ends, index, text
            )
            entity.add_parser(self.__class__)
            entities.append(entity)

        return entities

    def collect(
            self, token_list: List[Dict], index: int, text: str
    ) -> List[BaseEntity]:
        """
        Collect tokens where entity label is not O (viz outside in BIO tag format).

        Tags (BIO tagging) - https://natural-language-understanding.fandom.com/wiki/Named_entity_recognition#BIO

        Args:
            token_list (List[Dict]): XLMR NER produces this output.

        Returns:
            List[BaseEntity]
        """
        # To measure the string range of each token starting from the first.
        tape = 0
        entity_groups = {}

        # An example of `token_list` is:
        #
        # [{'I': 'O'},
        #   {'need': 'O'},
        #   {'a': 'O'},
        #   {'flight': 'B-vehicle'},
        #   {'from': 'O'},
        #   {'delhi': 'B-location'},
        #   {'to': 'O'},
        #   {'bangalore': 'B-location'}]
        #
        # Produced by: "I need a flight from delhi to bangalore"

        for token in token_list:
            # since the output is a `Dict[str, str]`, where the key is the token,
            for key, raw_type in token.items():
                # We check if the token is a valid entity that we want from the workflow.
                # Ideally "O" tagged items shouldn't be desired.
                if raw_type != "O":
                    # A measuring tape implementation, tape measures each token length and (+ 1) space.
                    # adds back to the `start` to measure span of new tokens.
                    entity_type = raw_type[2:]
                    start = tape
                    end = start + len(key)

                    if entity_type not in entity_groups:
                        entity_groups[entity_type] = [(raw_type, (key, start, end))]
                    else:
                        entity_groups[entity_type].append((raw_type, (key, start, end)))

                tape += len(key) + 1

        return self.combine_entity_groups(entity_groups, index, text)

    def entity_consensus(self, entities_list: List[Any]) -> List[Any]:
        """
        Resolve output from multiple sources.

        Args:
            items_list (List[Any]): List of list of intents or entities.
            type_ (str): One of intent or entities.

        Returns:
            List[Any]: List of intents or entities.
        """
        prevalent_entities = []
        entities_flattened = py_.flatten(entities_list)
        entity_groups = py_.group_by(entities_flattened, lambda entity: entity.type)
        for _, entities in entity_groups.items():
            if len(entities) > observations * threshold:
                entity = py_.min_by(entities, lambda entity: entity.alternative_index)
                prevalent_entities.append(entity)
        return prevalent_entities

    def extract(self, texts: List[str]) -> List[BaseEntity]:
        """
        NER from a given sentence.

        # Tags (BIO tagging) - https://natural-language-understanding.fandom.com/wiki/Named_entity_recognition#BIO
        **WARNING**: The implementation here doesn't ship a parser or a combining logic yet.

        Returns:
            List[BaseEntity]: List of entities
        """
        if not self.config.use_ner:
            return []

        # The second value is `raw_output` which can be used for estimating confidence
        # scores for each token identified as an entity.
        raw_token_lists, _ = self.ner.predict(texts)
        log.debug("raw_token_lists")
        log.debug(raw_token_lists)

        entities = []

        # An example of `raw_token_lists` is:
        #
        # [[{'I': 'O'},
        #   {'need': 'O'},
        #   {'a': 'O'},
        #   {'flight': 'B-vehicle'},
        #   {'from': 'O'},
        #   {'delhi': 'B-location'},
        #   {'to': 'O'},
        #   {'bangalore': 'B-location'}]]
        #
        # Produced by: "I need a flight from delhi to bangalore"

        for i, token_list in enumerate(raw_token_lists):
            entities.append(self.collect(token_list, i, texts[i]))

        return self.entity_consensus(entities)

    def inference(self):
        classifier_input = self.input[const.S_CLASSIFICATION_INPUT]
        ner_input = self.input[const.S_NER_INPUT]
        intent = self.classify(classifier_input)
        ner_entities = self.extract(ner_input)

        if self.output is not None:
            _, pre_filled_entities = self.output
            self.output = (intent, [*pre_filled_entities, *ner_entities])
        else:
            self.output = (intent, ner_entities)
