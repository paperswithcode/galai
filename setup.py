from setuptools import setup, find_packages

PACKAGE_NAME = 'galai'
VERSION = "1.1.6"
DESCRIPTION = "API for the GALACTICA model"
KEYWORDS = "Scientific Intelligence"
URL = 'https://github.com/paperswithcode/galai'
AUTHOR = 'Papers with Code'
LICENSE = 'Apache License 2.0'
REQUIRES_PYTHON = '>=3.7.0'
EXTRAS = {}

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f.readlines()]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=KEYWORDS,
    license=LICENSE,
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(include=f"{PACKAGE_NAME}.*"),
    install_requires=requirements,
    extras_require=EXTRAS,
    include_package_data=True,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
