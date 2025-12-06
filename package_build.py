#!/usr/bin/env python
import tomllib
import requests
from packaging.version import Version
import sys
import subprocess

def build_package(require_new_version=True):
    PACKAGE_NAME = "netlite"

    # 1. Read local version from pyproject.toml
    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    local_version = pyproject["project"]["version"]

    # 2. Get latest version from PyPI
    url = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"
    pypi_version = requests.get(url, timeout=10).json()["info"]["version"]

    # 3. Compare
    print("version in pyproject.toml: ", local_version)
    print("version on pypi.org:       ", pypi_version)
    if Version(local_version) <= Version(pypi_version) and require_new_version:
        print("Error: Local version must be increased for a new build")
        return False

    subprocess.run(['rm', '-rf', 'dist'])
    subprocess.run(['python', '-m', 'build'])

    return True
    
if __name__ == '__main__':
    build_package(require_new_version=False)
