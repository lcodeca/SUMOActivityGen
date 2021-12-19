#!/bin/bash
set -e

python3 -m pytest \
    --cov-report=term --cov-report=xml:coverage.xml \
    --cov-fail-under=90 \
    --cov-branch \
    --cov=saga/src saga/tests 