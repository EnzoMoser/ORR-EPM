#!/usr/bin/env bash
# This script sources the virtual python environment in private and adds the folders in this directory to path.

SOURCE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PYTHONPATH=$SOURCE_DIR
