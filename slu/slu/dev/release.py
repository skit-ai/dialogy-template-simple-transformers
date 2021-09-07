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
"""
import argparse
import os
import shutil
from datetime import datetime
from glob import glob

import semver
import toml
from dvc.repo import Repo as DVCRepo
from git import Actor, Repo
from git.refs.tag import TagReference
from prompt_toolkit import HTML, print_formatted_text, prompt

from slu import constants as const
from slu.dev.version import check_version_save_config
from slu.utils import logger
from slu.utils.config import Config, YAMLLocalConfig


def update_project_version_toml(version: str) -> None:
    """
    Update the version in pyproject.toml.

    Args:
        version (str): Current semver.
    """
    logger.debug("Updating pyproject.toml")
    project_toml_path = const.PROJECT_TOML

    with open(project_toml_path, "r") as toml_handle:
        toml_content = toml.load(toml_handle)

    toml_content[const.TOOL][const.POETRY][const.VERSION] = version

    with open(project_toml_path, "w") as toml_handle:
        toml.dump(toml_content, toml_handle)


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


def vcs(repo: Repo, version: str, changelog_body: str, active_branch: str) -> None:
    """
    Run dvc/git commands.
    """
    dvc_repo = DVCRepo()
    dvc_repo.add("data")

    # Stage
    index = repo.index
    index.add(
        [
            "data.dvc",
            const.PROJECT_TOML,
            const.CHANGELOG,
            os.path.join("config", "config.yaml"),
        ]
    )
    last_commit_author: Actor = repo.head.commit.author
    logger.info(f"Using {last_commit_author} for creating commits.")

    # Commit
    index.commit(
        f"update: {changelog_body}",
        author=last_commit_author,
        committer=last_commit_author,
    )

    # Tag version
    tag = repo.create_tag(version, message=f"{changelog_body}")

    # Push changes and tag
    logger.info(f"Pushing data to dvc.")
    dvc_repo.push()
    remote = repo.remote()
    logger.info(f"Pushing code to origin {active_branch}.")
    remote.push()
    logger.info(f"Pushing {tag} to origin.")
    remote.push(tag)


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

    with open(const.CHANGELOG, "r+") as changelog_handle:
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
    project_config_map = YAMLLocalConfig().generate()
    config: Config = list(project_config_map.values()).pop()
    check_version_save_config(config, version)

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

    # Remove everything except the current version, meant for release.
    remove_older_data_versions(version)

    # Update pyproject.toml to contain the release version.
    update_project_version_toml(version)

    # Maintain changelog.
    changelog_body = update_changelog(version)

    # version control commands
    # git and dvc add, commit, push combo.
    # -----------------------------------------------------
    vcs(repo, version, changelog_body, active_branch)
    # -----------------------------------------------------
