#!/bin/bash

# SUMO Activity-Based Mobility Generator - Linting check
#
# Author: Lara CODECA
#
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# http://www.eclipse.org/legal/epl-2.0.

# exit on error
set -e

# Linting everything
black . --check

# Reordering the imports
isort . --check

# Lint for all the python files
pylint --rcfile=.pylintrc $(find . -name '*.py')