from multiprocessing import Pool

import numpy as np
import pandas as pd  # type: ignore

from slu import constants as const


def parallel_proc(
    items, fn, calibration_config, return_df=False, n_cores=const.N_DEFAULT_CORES
):
    """
    Perform arbitrary functions of a dataframe.

    Using multi-processing on a dataframe to preprocess records.
    """

    def wrapper(items):
        return fn(items, calibration_config=calibration_config)

    if n_cores == const.N_MIN_CORES:
        processed_items = fn(items, calibration_config)
    else:
        chunks = np.array_split(items, n_cores)
        pool = Pool(n_cores)
        if return_df:
            processed_items = pd.concat(pool.map(wrapper, chunks))
        else:
            processed_items = pool.map(wrapper, chunks)
        pool.close()
    return processed_items
