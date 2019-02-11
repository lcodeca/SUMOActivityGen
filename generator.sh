#!/bin/bash

# SUMO Activity-Based Mobility Generator
#     Copyright (C) 2019
#     EURECOM - Lara CODECA

# exit on error
set -e

if [ -z "$MOBILITY_GENERATOR" ]
then
    echo "Environment variable MOBILITY_GENERATOR is not set."
    echo "Please set MOBILITY_GENERATOR to the root directory."
    echo "Bash example:"
    echo "      in MoSTScenario exec"
    echo '      export MOBILITY_GENERATOR=$(pwd)'
    exit
fi

SCENARIO="$MOBILITY_GENERATOR/scenario"
INPUT="$SCENARIO/sumofiles"
ADD="$INPUT/add"

OUTPUT="out"
mkdir -p $OUTPUT

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PUBLIC TRANSPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

# https://github.com/eclipse/sumo/issues/3803
#  Depending on the SUMO version, it's possible that the -b parameter is not really working.

# INTERVAL="-b 14400 -e 50400"
INTERVAL="-b 0 -e 86400"

echo "[$(date)] --> Generate bus trips..."
python $SUMO_TOOLS/ptlines2flows.py -n $INPUT/most.net.xml $INTERVAL -p 900 \
    --random-begin --seed 42 --no-vtypes \
    --ptstops $ADD/most.busstops.add.xml --ptlines $SCENARIO/pt/most.buslines.add.xml \
    -o $OUTPUT/most.buses.flows.xml

sed -e s/:0//g -i $OUTPUT/most.buses.flows.xml

echo "[$(date)] --> Generate train trips..."
python $SUMO_TOOLS/ptlines2flows.py -n $INPUT/most.net.xml $INTERVAL -p 1200 \
    -d 300 --random-begin --seed 42 --no-vtypes \
    --ptstops $ADD/most.trainstops.add.xml --ptlines $SCENARIO/pt/most.trainlines.add.xml \
    -o $OUTPUT/most.trains.flows.xml

sed -e s/:0//g -i $OUTPUT/most.trains.flows.xml

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TRACI MOBILITY GENERATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

echo "[$(date)] --> Generate mobility..."
python3 activitygen.py -c activitygen.json