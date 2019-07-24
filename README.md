# SUMOActivityGen

Activity-based Mobility Generation for SUMO Simulator

Contacts: Lara CODECA [lara.codeca@gmail.com], Jerome HAERRI [haerri@eurecom.fr]

This program and the accompanying materials are made available under the
terms of the Eclipse Public License 2.0 which is available at http://www.eclipse.org/legal/epl-2.0.

## Overview

This project is in its early stages, and it is still under development.

* It can be tested and explored using the configuration files provided by [MoSTScenario](https://github.com/lcodeca/MoSTScenario) and `bash most.generator.sh`
* The complete generation of a scenario from OSM can be done using `python3 scenarioFromOSM.py --osm {osm file} --out {putput directory}`. All the generated files are going to be in the output directory.

![SUMOActivityGen Overview](https://github.com/lcodeca/SUMOActivityGen/blob/master/SUMOActivityGen.png)

![Scenario Generation Overview](https://github.com/lcodeca/SUMOActivityGen/blob/master/ScenarioGenerator.png)

### Due to some changes in the SUMO development version of the TraCI APIs, the master branch is not compatible with SUMO 1.2.0

* _Release v0.1 is compatible with SUMO 1.2.0_

## HOW TO

### Required libraries

`pip3 install tqdm pyproj numpy shapely matplotlib rtree`
To use 'rtree', 'libspatialindex-dev' is requiresd to be installed.

### The SUMOActivityGen mobility generator

`python3 activitygen.py -c configuration.json`

### The Scenario Generation from OSM

`python3 scenarioFromOSM.py --osm file.osm --out target_directory`

Optional parameters:

```
  --lefthand                Generate a left-hand traffic scenario.
  --population POPULATION   Number of people plans to generate.
  --density DENSITY         Average population density in square kilometers.
  --single-taz              Ignore administrative boundaries and generate only one TAZ.
  --from-step FROM_STEP     For successive iteration of the script,
                            it defines from which step it should start:
                            [0 - Copy default files.]
                            [1 - Run netconvert & polyconvert.]
                            [2 - Run ptlines2flows.py.]
                            [3 - Generate parking areas.]
                            [4 - Generate parking area rerouters.]
                            [5 - Extract TAZ from administrative boundaries.]
                            [6 - Generate OD-matrix.]
                            [7 - Generate SUMOActivityGen defaults.]
                            [8 - Run SUMOActivityGen.]
                            [9 - Launch SUMO.]
  --profiling               Enable Python3 cProfile feature.
  --no-profiling            [default] Disable Python3 cProfile feature.
```
