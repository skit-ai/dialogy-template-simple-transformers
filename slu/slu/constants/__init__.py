"""
Add symbols that can be used as project level constants.
Makes for easy refactoring in case of renames, typos and keeping values consistent.
"""
import os

N_DEFAULT_CORES = 8
N_MIN_CORES = 1
N_EVAL_BATCH_SIZE = 8
N_EVAL_SPLIT = 0.1
N_EPOCHS = 10
k = 1000

NAME = "name"
MODEL_NAME = "model_name"
SLU = "slu"
CORES = "cores"
DATA_ID = "data_id"
DATA = "data"
TAG = "tag"
ALTERNATIVES = "alternatives"
DATASETS = "datasets"
MODELS = "models"
METRICS = "metrics"
INTENT = "intent"
INTENT_PRED = "intent_pred"
INTENTS = "intents"
URL = "url"
ENTITIES = "entities"
INPUT = "input"
OUTPUT = "output"
TYPE = "type"
PARSER = "parser"
SENTENCE_ID = "sentence_id"
WORDS = "words"
LABELS = "labels"
REPORT = "report"
VERSION = "version"
TRAIN = "train"
TEST = "test"
SKIPPED = "skipped"
DEFAULT = "default"
DEV = "dev"
EPOCHS = "epochs"
CLASSIFICATION = "classification"
NER = "ner"
TASKS = "tasks"
ALIAS = "alias"
THRESHOLD = "threshold"
RULES = "rules"
SLOTS = "slots"
TOOL = "tool"
POETRY = "poetry"
MASTER = "master"
CONTEXT = "context"
TRANSCRIPT = "transcript"
TEXT = "text"
CONFIDENCE = "confidence"
INIT = "init"
CLONE = "clone"
REPL = "repl"
LANGUAGES = "languages"
LANGUAGE = "language"
RELEASE = "release"
CSV = "csv"
SQLITE = "sqlite"
USE = "use"
FORMAT = "format"
METADATA = "metadata"
ENVIRONMENT = "ENVIRONMENT"
PRODUCTION = "production"
STAGING = "staging"
TRUE_LABEL = "true_label"
PRED_LABEL = "pred_label"
VALUE = "value"
VALUES = "values"
PLUGIN = "plugin"
PARAMS = "params"
RULE_BASED_SLOT_FILLER_PLUGIN = "RuleBasedSlotFillerPlugin"
DUCKLING = "Duckling"
DUCKLING_PLUGIN = "DucklingPlugin"
LIST_ENTITY_PLUGIN = "ListEntityPlugin"
CANDIDATES = "candidates"
STYLE = "style"
REGEX = "regex"
ENTITY_MAP = "entity_map"
PROJECT_NAME = "project_name"
COMMON = "common"
OUTPUT_DIR = "output_dir"
BEST_MODEL_DIR = "best_model_dir"
EXTEND = "extend"
REPLACE = "replace"
HISTORY = "history"
REWIND = "rewind"
FORWARD = "forward"
RANGE = "range"
START = "start"
END = "end"
PARSERS = "parsers"

TRAIN_DATA = "train.csv"
TEST_DATA = "test.csv"
CONFIG_PATH = os.path.join("config", "config.yaml")
PROMPTS_CONFIG_PATH = os.path.join("config", "prompts.yaml")
MISSING_PROMPTS_PATH = os.path.join("config", "missing_prompts.yaml")
INTENT_LABEL_ENCODER = "labelencoder.pkl"
ENTITY_LABELS = "entity_label_list.pkl"
XLMR = "xlmroberta"
XLMRB = "xlm-roberta-base"
REPORT = "report.csv"
CLASSIFICATION_TASK = "intent_classification"
EXTRACTION_TASK = "entity_extraction"
BEST_MODEL = "best_model_dir"
OUTPUT_DIR = "output_dir"
EVAL_DURING_TRAINING_STEPS = "evaluate_during_training_steps"
EVAL_BATCH_SIZE = "eval_batch_size"
MODEL_ARGS = "model_args"
NUM_TRAIN_EPOCHS = "num_train_epochs"
PROJECT_TOML = "pyproject.toml"
CHANGELOG = "CHANGELOG.md"

INTENT_ERROR = "_error_"
INTENT_OOS = "_oos_"
CLASSIFICATION_INPUT = "classification_input"
NER_INPUT = "ner_input"
CONTEXT = "context"
ERRORS = "errors.csv"
INTENTS_INFO = "intents_info"
REFERENCE_TIME = "reference_time"
LOCALE = "locale"
CURRENT_STATE = "current_state"
CURRENT_INTENT = "current_intent"
STATE = "state"
EXPECTED_SLOTS = "expected_slots"
NLS_LABEL = "nls_label"
PROMPT_NOISE_FILLER_TOKEN = "<pad>"
LANG_TO_LOCALES = {                     
    "en": "en_IN", 
    "hi": "hi_IN"
}                                        # This should be set via config
PLATFORM_LEVEL_NOISE = {                 # Mapping between accepted lang formats and their respective platform-level variants (non-standard/noise).  
    "en": ["en_nls", "en_us"],           # Type: Dict[str, list]; Each lang has multiple noisy variants. 
    "hi": ["hi_nls","hi_us"]
}
CLIENTCONFIGROUTE = "/clients/configs/"
REQUEST_MAX_RETRIES = 5
LANG = "lang"
TEXTS_TO_IGNORE = {"<UNK>"}
SCORE = "score"
REFERENCE_TIME = "reftime"
CONFIG = "config"
CUDA_WITH_DYNAMIC_QUANTIZATION_MESSAGE = (
    "Could not run 'quantized::linear_dynamic' with arguments from the 'CUDA' backend"
)

LOW = "low"
MEDIUM = "medium"
HIGH = "high"
CONFIDENCE_LEVEL = "confidence_levels"

POD_RESTART_MSG_THRESHOLD = 3
POD_FAILURE_MSG_THRESHOLD = 1
POD_HASH_LEN = 17
NOTIFICATION_COOLDOWN = 20
MAX_NOTIFS = 3
