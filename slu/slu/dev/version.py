import semver
from slu.utils import logger


def check_version_save_config(config, version):
    if version:
        semver.VersionInfo.parse(version)
        logger.info(f"Using version: {version}.")
        config.version = version
        config.save()
