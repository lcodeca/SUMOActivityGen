#!/bin/bash
set -e

ROOT=$(pwd)

# Install SAGA
cd saga 
python3 -m pip install -e . 
cd ..

export SINGLETAZ='--single-taz'
export TAZLVL9='--admin-level 9'
export COMMON='--local-defaults --processes 4 --population 1000 --taxi-fleet 100'

rm -rf local_tests
mkdir local_tests
cp test_osm/* local_tests/. 
gunzip local_tests/*.gz

echo "Running DLR Scenario - Germany"
time python3 scenarioFromOSM.py --osm local_tests/dlr_osm.xml --out local_tests/dlr $COMMON $SINGLETAZ

echo "Running Docklands - Ireland"
time python3 scenarioFromOSM.py --osm local_tests/docklands_osm.xml --out local_tests/docklands --lefthand $COMMON $TAZLVL9

echo "Running Sophia Antipolis - France"
time python3 scenarioFromOSM.py --osm local_tests/sophia_osm.xml --out local_tests/sophia $COMMON $TAZLVL9

echo "Running Riyadh - Saudia Arabia"
time python3 scenarioFromOSM.py --osm local_tests/riyadh_osm.xml --out local_tests/riyadh $COMMON $TAZLVL9

echo "Running Kyoto - Japan"
time python3 scenarioFromOSM.py --osm local_tests/kyoto_osm.xml --out local_tests/kyoto --lefthand $COMMON $TAZLVL9
