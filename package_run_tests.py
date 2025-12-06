#!/usr/bin/env python
import subprocess
import sys

def run_tests():
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"]) # ensures python interpreter in venv
    return result.returncode == 0

if __name__ == '__main__':
    success = run_tests()
    print("Tests passed:", success)
