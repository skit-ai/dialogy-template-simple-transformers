import functools

from slu import constants as const


def task_guard(func):
    @functools.wraps
    def wrapper(self, task_name: str, *args, **kwargs):
        supported_tasks = {const.CLASSIFICATION, const.NER}

        if task_name not in supported_tasks:
            raise ValueError(f"Task should be one of {supported_tasks}")

        use_task = self.task_by_name(task_name).use

        if use_task:
            value = func(task=task_name,*args, **kwargs)
            return value
        else:
            raise ValueError(f"Attempting to run a {task_name} task but {func} failed,"
            f" because config has {const.TASKS}.")
    return wrapper
