#!/bin/bash

# SUMO Activity-Based Mobility Generator
#
# Copyright (c) 2019 Lara CODECA - EURECOM
#
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# http://www.eclipse.org/legal/epl-2.0.

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

SCENARIO="$MOBILITY_GENERATOR/MoSTScenario/scenario"
MOBILITY_TOOLS="$MOBILITY_GENERATOR/MoSTScenario/tools/mobility"

INPUT="$SCENARIO/in"
ADD="$INPUT/add"

OUTPUT="rou"
mkdir -p $OUTPUT

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PUBLIC TRANSPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

INTERVAL="-b 0 -e 86400"

echo "[$(date)] --> Generate bus trips..."
python $SUMO_TOOLS/ptlines2flows.py -n $INPUT/most.net.xml $INTERVAL -p 900 \
    --random-begin --seed 42 --no-vtypes \
    --ptstops $ADD/most.busstops.add.xml --ptlines $MOBILITY_TOOLS/pt/most.buslines.add.xml \
    -o $OUTPUT/most.example.buses.flows.xml

sed -e s/:0//g -i $OUTPUT/most.example.buses.flows.xml

echo "[$(date)] --> Generate train trips..."
python $SUMO_TOOLS/ptlines2flows.py -n $INPUT/most.net.xml $INTERVAL -p 1200 \
    -d 300 --random-begin --seed 42 --no-vtypes \
    --ptstops $ADD/most.trainstops.add.xml --ptlines $MOBILITY_TOOLS/pt/most.trainlines.add.xml \
    -o $OUTPUT/most.example.trains.flows.xml

sed -e s/:0//g -i $OUTPUT/most.example.trains.flows.xml

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TRACI MOBILITY GENERATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

echo "[$(date)] --> Generate mobility..."
python3 activitygen.py -c most.activitygen.json

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SUMO SIMULATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

echo "[$(date)] --> Running the SUMO simulation..."
sumo -c most.test.sumocfg