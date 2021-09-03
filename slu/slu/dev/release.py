"""
This module offers utilities to version datasets, models, metrics and related code.

Boilerplate sugar for:

1. dvc pull
2. rm -rf data/<version_others> ; if <version> not in <version_others>
3. dvc add data
4. git add data.dvc
5. <version> >> pyproject.toml
6. git commit -m "${message}"
7. git tag <semver> "${message}"
8. git push origin <semver>
9. dvc push

To ensure this works correctly, we need to check:

1. There are no uncommitted / unstaged changes.
2. dvc is initialized.
3. dvc remote is configured.
4. A tag with the same version doesn't already exist.
"""
import os
import shutil
import argparse
import subprocess
from configparser import ConfigParser
from datetime import datetime
from glob import glob

import semver
import toml
from git import Repo
from git.refs.tag import TagReference
from prompt_toolkit import HTML, print_formatted_text, prompt

from slu import constants as const
from slu.utils.config import Config
from slu.utils import logger


def update_project_version_toml(version: str) -> None:
    """
    Update the version in pyproject.toml.

    Args:
        version (str): Current semver.
    """
    logger.debug("Updating pyproject.toml")
    project_toml_path = const.S_PROJECT_TOML

    with open(project_toml_path, "r") as toml_handle:
        toml_content = toml.load(toml_handle)

    toml_content[const.TOOL][const.POETRY][const.VERSION] = version

    with open(project_toml_path, "w") as toml_handle:
        toml.dump(toml_content, toml_handle)


def update_config(version: str) -> None:
    """
    Update config.yaml with the latest version.

    Args:
        version (str): Current semver.
    """
    config = Config()
    config.set_version(version)
    config.save()


def remove_older_data_versions(current_version: str) -> None:
    """
    Remove versions that are older than current release.

    Pushing all datasets to master/tag would make it very inconvenient to access
    as datasets and models are huge downloads. To go easy on the bandwidth tags and master branch
    will only have the latest version and older ones are expected to be accessible using their tags.

    Args:
        current_version (str): Current semver version, versions other than this would be removed.
    """
    for version_module in glob(os.path.join(const.DATA, "**")):
        if current_version not in version_module:
            logger.debug("Removing %s to keep the tag light.", version_module)
            shutil.rmtree(version_module)


def vcs_add() -> None:
    """
    Run dvc/git commands in a dirty way.
    """
    subprocess.call(["dvc", "add", "data"], shell=False)
    subprocess.call(["git", "add", "data.dvc"], shell=False)
    subprocess.call(["git", "add", "CHANGELOG.md"], shell=False)
    subprocess.call(["git", "add", "pyproject.toml"], shell=False)
    subprocess.call(["git", "add", os.path.join("config", "config.yaml")], shell=False)


def vcs_tag_and_commit_state(version, changelog_body) -> None:
    """
    Update repository state with tag=version.
    """
    message = f"update: {changelog_body}"
    subprocess.call(["git", "commit", "-m", message], shell=False)
    subprocess.call(["git", "tag", "-a", version, "-m", message], shell=False)


def vcs_push_remote(version: str, branch: str) -> None:
    """
    Updte repository remote with local branch updates.

    Args:
        version (str): current semver.
        branch (str): current active branch.
    """
    subprocess.call(["git", "push", "origin", version], shell=False)
    subprocess.call(["git", "push", "origin", branch], shell=False)
    subprocess.call(["dvc", "push"], shell=False)


def is_dvc_remote_set() -> bool:
    """
    Check if dvc config exists and s3 remote is set.

    Returns:
        bool: True if .dvc/config has s3remote configured.
    """
    dvc_config_path = os.path.join(".dvc", "config")
    if not os.path.exists(dvc_config_path):
        return False

    config = ConfigParser()
    _ = config.read(dvc_config_path)
    sections = config.sections()

    if "'remote \"s3remote\"'" not in sections:
        return False

    return True


def update_changelog(version: str) -> str:
    """
    Read user input and update changelog.

    Read multiline markdown friendly changelog data and log in CHANGELOG.md.

    Args:
        version (str): semver.

    Returns:
        str: Changelog content.
    """
    separator = "-" * 50 + "\n"
    print_formatted_text(
        HTML(
            f"What makes version {version} remarkable? "
            "\nThis input is <b><ansigreen>markdown friendly</ansigreen></b> and <b><ansigreen>date</ansigreen></b> will be added automatically!\n(Press ESC"
            " followed by ENTER to submit)\n" + separator
        )
    )

    raw_changelog = prompt("", multiline=True)

    timestamp = datetime.strftime(datetime.now(), "%A, %d %B %Y | %I:%M %p")
    changelog_body = raw_changelog.strip()
    changelog = f"# {version} | {timestamp}\n\n{changelog_body}"

    with open(const.S_CHANGELOG, "r+") as changelog_handle:
        previous_logs = changelog_handle.read().strip()
        changelog_handle.seek(0, 0)
        content = changelog + "\n\n" + previous_logs
        changelog_handle.write(content.strip())
    return changelog_body


def release(args: argparse.Namespace) -> None:
    """
    Update data directory via version control utils.

    Boilerplate for usual but tedious version control steps.

    Args:
        version (str): Semver for the dataset, model and metrics.
    """
    version = args.version
    # Ensure `version` is a valid semver.
    semver.VersionInfo.parse(version)

    # Interact with the git repo, assumes the script root contains the repo.
    repo = Repo(".")

    # Check for unstaged or uncommitted changes.
    if repo.is_dirty():
        logger.error("There are unstaged / uncommitted changes.")
        return None

    active_branch = repo.active_branch.name

    # Fetch list of tags from the remote, to prevent creating tags that already exist.
    for remote in repo.remotes:
        remote.fetch()

    tags = [tag.name for tag in TagReference.list_items(repo)]
    if version in tags:
        logger.error(
            "Version %s already exists. Use `git tag` or `git tag -l %s` to verify.",
            version,
            version,
        )
        return None

    if not is_dvc_remote_set():
        logger.error(
            "Looks like dvc remote is not set. We won't be able to push code."
            "\nRun:\n\ndvc remote add -d s3remote s3://bucket_name/path/to/dir\n\n... to use this command."
        )
        return None

    # Remove everything except the current version, meant for release.
    remove_older_data_versions(version)

    # Update pyproject.toml to contain the release version.
    update_project_version_toml(version)

    # Update config/config.yaml to contain the release version.
    update_config(version)

    # Maintain changelog.
    changelog_body = update_changelog(version)

    # version control commands
    # git and dvc add, commit, push combo.
    # -----------------------------------------------------
    vcs_add()
    vcs_tag_and_commit_state(version, changelog_body)
    vcs_push_remote(version, active_branch)
    # -----------------------------------------------------
