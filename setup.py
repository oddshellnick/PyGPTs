import os

from setuptools import find_packages, setup


def get_long_description() -> str:
    lib_folder = os.path.dirname(os.path.realpath(__file__))
    long_description_path = f"{lib_folder}/long_description.rst"

    return open(long_description_path, "r", encoding="utf-8").read()


def get_install_requires() -> list[str]:
    lib_folder = os.path.dirname(os.path.realpath(__file__))
    requirement_path = f"{lib_folder}/requirements.txt"

    install_requires = []

    if os.path.isfile(requirement_path):
        install_requires = open(requirement_path, "r", encoding="utf-8").read().splitlines()

    return install_requires


def get_description() -> str:
    lib_folder = os.path.dirname(os.path.realpath(__file__))
    description_path = f"{lib_folder}/description.txt"

    return open(description_path, "r", encoding="utf-8").read()


setup(
    name="PyGPTs",
    version="1.0.7",
    author="oddshellnick",
    author_email="oddshellnick.programming@gmail.com",
    description=get_description(),
    long_description=get_long_description(),
    long_description_content_type="text/x-rst",
    packages=find_packages(),
    install_requires=get_install_requires(),
)
