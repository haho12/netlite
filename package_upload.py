#!/usr/bin/env python
import tomllib
import requests
from packaging.version import Version
import sys
import subprocess

from package_run_tests import run_tests
from package_build import build_package

PACKAGE_NAME = "netlite"

# Steps:
# - run pytest
# - build (with new release number)
# - push to github (with release tag)
# - upload to pipy

def git_tag_and_push():
    # Check code is fully committed in git
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True
    ).stdout.strip()

    if status:
        print("Error: Working tree is dirty. Commit or stash your changes first.")
        print(status)
        return False

    subprocess.run(['git', 'push'])                # push commits
        
    # 1. Read local version from pyproject.toml
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    local_version = pyproject["project"]["version"]

    # 2. Get latest version from PyPI
    #url = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
    #pypi_version = requests.get(url, timeout=10).json()["info"]["version"]

    # 3. Get latest git tag
    git_tag = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"],
        capture_output=True,
        text=True,
        check=True
    ).stdout.strip()
    # strip leading "v" (v1.2.3 → 1.2.3)
    git_version = git_tag.lstrip("v")

    # 4. Compare
    print("version in pyproject.toml: ", local_version)
    print("version in git tag:        ", git_version)
    #print("version on pypi.org:       ", pypi_version)
    if Version(git_version) < Version(local_version):
        git_tag_new = 'v' + local_version
        print(f"Info: Updating git tag to '{git_tag_new}'")
        subprocess.run(['git', 'tag', git_tag_new])
        subprocess.run(['git', 'push', git_tag_new])   # push new tag

        sys.exit()

success = run_tests()
if not success:
    sys.exit()

success = build_package(require_new_version=True)
if not success:
    sys.exit()

success = git_tag_and_push()
if not success:
    sys.exit()

#subprocess.run(['twine', 'upload', 'dist/*'])