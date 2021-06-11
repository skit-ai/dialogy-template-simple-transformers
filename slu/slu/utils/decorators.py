import functools

from slu import constants as const

def checks_usage(func):
    @functools.wraps
    def wrapper(self, task, *args, **kwargs):
        if (task == const.CLASSIFICATION and self.use_classification or
            task == const.NER and self.use_ner):
            value = func(task=task,*args, **kwargs)
            return value
    return wrapper
