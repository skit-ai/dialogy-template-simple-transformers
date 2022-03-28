import asyncio
from functools import wraps, partial

from slu import constants as const


def task_guard(func):
    def wrapper(self, task_name: str, *args, **kwargs):
        supported_tasks = {const.CLASSIFICATION, const.NER}

        if task_name not in supported_tasks:
            raise ValueError(f"Task should be one of {supported_tasks}")

        use_task = self.task_by_name(task_name).use

        if use_task:
            value = func(self, task_name, *args, **kwargs)
            return value
        else:
            return None

    return wrapper
    

def async_wrap(func):
    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, pfunc)
    return run