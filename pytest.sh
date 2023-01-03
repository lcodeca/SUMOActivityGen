#!/bin/bash
set -e

# Currently there are not tests written, so --cov-fail-under=99 is not yet implemented.

python3 -m pytest \
    --cov-report=term --cov-report=xml:coverage.xml \
    --cov-branch \
    --cov=saga/src saga/tests 