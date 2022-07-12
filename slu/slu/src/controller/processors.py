import os
from typing import List, Optional

from dialogy import plugins
from dialogy.base.plugin import Plugin
from dialogy.workflow import Workflow

from slu import constants as const
from slu.utils.config import load_gen_config, load_prompt_config, Config
from slu.src.controller.custom_plugins import OOSFilterPlugin


class SLUPipeline:
    def __init__(self, config: Optional[Config]=None, **kwargs):
        self.config = config or kwargs.get(const.CONFIG) or load_gen_config()
        self.debug = kwargs.get("debug", False)
        self.prompts_map = load_prompt_config()

    def get_plugins(self, purpose) -> List[Plugin]:
        merge_asr_output = plugins.MergeASROutputPlugin(
            dest="input.clf_feature",
            use_transform=True,
            input_column=const.ALTERNATIVES,
            debug=self.debug,
        )
        
        duckling_plugin = plugins.DucklingPlugin(
            dest="output.entities",
            dimensions=["people", "number", "time", "duration"],
            locale="en_IN",
            timezone="Asia/Kolkata",
            timeout=0.5,
            input_column=const.ALTERNATIVES,
            output_column=const.ENTITIES,
            constraints=self.config.timerange_constraints,
            # url works only in development mode.
            # You need to set its real value in k8s configs or wherever you keep your
            # env-vars safe.
            url=os.environ.get("DUCKLING_URL", "http://localhost:8000/parse/"),
            use_transform=False,
            debug=self.debug,
        )

        list_entity_plugin = plugins.ListEntityPlugin(
            dest="output.entities",
            style=const.REGEX,
            candidates=self.config.entity_patterns,
            threshold=0.1,
            input_column=const.ALTERNATIVES,
            output_column=const.ENTITIES,
            use_transform=False,
            debug=self.debug,
        )


        xlmr_clf = plugins.XLMRMultiClass(
            dest="output.intents",
            model_dir=self.config.get_model_dir(const.CLASSIFICATION),
            score_round_off=5,
            purpose=purpose,
            use_cuda=purpose != const.PRODUCTION,
            data_column=const.ALTERNATIVES,
            label_column=const.TAG,
            args_map=self.config.get_model_args(const.CLASSIFICATION),
            debug=self.debug,
            state_column='state',
            lang_column='lang',
            prompts_map =  self.prompts_map,
            use_state = False,
            use_prompt=False,
            train_using_all_prompts=False
        )

        oos_filter = OOSFilterPlugin(
            dest="output.intents",
            threshold=self.config.get_model_confidence_threshold(const.CLASSIFICATION),
            replace_output=True,
            guards=[lambda i, o: purpose != const.PRODUCTION],
        )

        slot_filler = plugins.RuleBasedSlotFillerPlugin(
            dest="output.intents",
            rules=self.config.slots,
            debug=self.debug,
            fill_multiple=True,
        )

        return [merge_asr_output, xlmr_clf, oos_filter, slot_filler]

    @staticmethod
    def filter_plugins(all_plugins, final_plugin):
        """
        This will ensure that pipeline runs till final_plugin only and then returns.
        """
        for idx, plugin in enumerate(all_plugins):
            if isinstance(plugin, final_plugin):
                return all_plugins[:idx+1]
        return all_plugins
        

    def get_workflow(self, purpose, final_plugin=None):
        self.plugins = self.get_plugins(purpose)
        if final_plugin:
            self.plugins = self.filter_plugins(self.plugins, final_plugin)

        return Workflow(self.plugins, debug=self.debug)
