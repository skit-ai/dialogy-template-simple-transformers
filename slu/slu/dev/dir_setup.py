"""
This module contains utilities for creating a default data directory.
A data directory for this template contains the following structure:

```
data
|-- <version>
    |-- classification
    |   |-- datasets
    |   |-- metrics
    |   +-- models
```

Given a valid semver, the code here helps creating
and hence maintaining the uniformity of the directory structure here.
"""
import argparse
import os
import shutil

import semver

from slu import constants as const


def create_data_directory(args: argparse.Namespace) -> None:
    """
    Create sub directories.

    This function will check if `version` is a valid semver.

    Args:
        version (str): Semver for the dataset, model and metrics.
    """
    version = args.version

    # This will raise an exception for invalid semver. So we don't have to catch it.
    semver.VersionInfo.parse(version)

    base_module_path = os.path.join(const.DATA, version)
    depth_level_1 = [const.CLASSIFICATION]
    depth_level_2 = [const.DATASETS, const.METRICS, const.MODELS]

    for subdir in depth_level_1:
        for childdir in depth_level_2:
            os.makedirs(os.path.join(base_module_path, subdir, childdir), exist_ok=True)


def copy_data_directory(args: argparse.Namespace) -> None:
    """
    Copy subdirectory.

    1. This function will check `copy_from` and `copy_to` are valid semver.
    2. This function will check `copy_to` doesn't already exist.

    Args:
        copy_from (str): semver -> Source directory.
        copy_to (str): semver -> Destination directory.
    """
    copy_from = args.source
    copy_to = args.dest

    # This will raise an exception for invalid semver. So we don't have to catch it.
    semver.VersionInfo.parse(copy_from)
    semver.VersionInfo.parse(copy_to)

    source = os.path.join(const.DATA, copy_from)
    destination = os.path.join(const.DATA, copy_to)
    shutil.copytree(source, destination)
