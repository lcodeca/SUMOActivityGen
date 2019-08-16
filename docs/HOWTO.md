# HOW TO

## Required libraries

To use this software the following Python 3 libraries are required:

* `pip3 install tqdm pyproj numpy shapely matplotlib rtree`

To use `rtree`, `libspatialindex-dev` is required to be installed.

## The Scenario Generation from OSM

Documentation in [docs/ScenarioGeneration.md](ScenarioGeneration.md).

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

## The SUMOActivityGen mobility generator

Documentation in [docs/SUMOActivityGen.md](SUMOActivityGen.md).

`python3 activitygen.py -c configuration.json`
