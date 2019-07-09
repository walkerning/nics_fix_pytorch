# -*- coding: utf-8 -*-
import os
import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

here = os.path.dirname(os.path.abspath((__file__)))

# meta infos
NAME = "nics_fix_pytorch"
DESCRIPTION = "Fixed point trainig framework on PyTorch"
with open(os.path.join(os.path.dirname(__file__), "nics_fix_pt/VERSION")) as f:
    VERSION = f.read().strip()


AUTHOR = "foxfi"
EMAIL = "foxdoraame@gmail.com"

# package contents
MODULES = []
PACKAGES = find_packages(exclude=["tests.*", "tests"])

# dependencies
INSTALL_REQUIRES = ["six==1.11.0", "numpy"]
TESTS_REQUIRE = ["pytest", "pytest-cov", "PyYAML"]  # for mnist example

# entry points
ENTRY_POINTS = """"""


def read_long_description(filename):
    path = os.path.join(here, filename)
    if os.path.exists(path):
        return open(path).read()
    return ""


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ["tests/", "--junitxml", "unittest.xml", "--cov"]
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren"t loaded
        import pytest

        errno = pytest.main(self.test_args)
        sys.exit(errno)


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=read_long_description("README.md"),
    author=AUTHOR,
    author_email=EMAIL,
    py_modules=MODULES,
    packages=PACKAGES,
    entry_points=ENTRY_POINTS,
    zip_safe=False,
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    cmdclass={"test": PyTest},
    package_data={
        "nics_fix_pt": ["VERSION"]
    }
)
