#!/bin/bash

# python version
PYTHON=3.7.3

# automatically detected variables
VENV=$(basename $(dirname $(realpath $0)))

# validation
if ! type pyenv > /dev/null; then
    echo "pyenv not found."
    exit 1
fi

# install
echo "setup virtualenv"
pyenv virtualenv $PYTHON $VENV
pyenv local $VENV
pip install --upgrade pip

echo "install submodules"
git submodule init
git submodule update --recursive

# rehash to enable command line integration
pyenv rehash
