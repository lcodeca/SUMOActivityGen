#!/bin/bash
set -e

# Set a Ubuntu Default for the SUMO_HOME is it's not already set
# It's useful for the buildenv
export SUMO_HOME="${SUMO_HOME:=/usr/share/sumo}"  # If variable not set or null, set it to default.


# Currently there are not tests written, so --cov-fail-under=99 is not yet implemented.

python3 -m pytest \
    --cov-report=term --cov-report=xml:coverage.xml \
    --cov-branch \
    --cov=saga/src saga/tests 