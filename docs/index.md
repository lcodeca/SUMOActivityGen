# HOW TO

## Requirements

### Libraries

To use this software the following Python 3 libraries are required:

* `pip3 install tqdm pyproj numpy shapely matplotlib rtree folium`

To use `rtree`, `libspatialindex-dev` is required to be installed.

### Eclipse SUMO

This software requires [Eclipse SUMO v1.3](https://github.com/eclipse/sumo/releases/tag/v1_3_0)

## The Scenario Generation from OSM

Documentation in [docs/ScenarioGeneration.md](ScenarioGeneration.md).

`python3 scenarioFromOSM.py --osm file.osm --out target_directory`

Optional parameters:

```
  --osm OSM_FILE        OSM file.
  --out OUT_DIR         Directory for all the output files.
  --lefthand            Generate a left-hand traffic scenario.
  --population POPULATION
                        Number of people plans to generate.
  --taxi-fleet TAXI_FLEET
                        Size of the taxi fleet.
  --density DENSITY     Average population density in square kilometers.
  --single-taz          Ignore administrative boundaries and generate only one
                        TAZ.
  --admin-level ADMIN_LEVEL
                        Select only the administrative boundaries with the
                        given level and generate the associated TAZs.
  --taz-plot HTML_FILENAME
                        Plots the TAZs to an HTML file as OSM overlay.
                        (Requires folium)
  --processes PROCESSES
                        Number of processes spawned (when suported) to
                        generate the scenario.
  --from-step FROM_STEP
                        For successive iteration of the script, it defines
                        from which step it should start:
                             0 - Copy default files.
                             1 - Run netconvert & polyconvert.
                             2 - Run ptlines2flows.py.
                             3 - Generate parking areas.
                             4 - Generate parking area rerouters.
                             5 - Generate taxi stands.
                             6 - Generate taxi stands rerouters.
                             7 - Extract TAZ from administrative boundaries.
                             8 - Generate OD-matrix.
                             9 - Generate SUMOActivityGen defaults.
                            10 - Run SUMOActivityGen.
                            11 - Launch SUMO.
                            12 - Report.
  --to-step TO_STEP     For successive iteration of the script, it defines
                        after which step it should stop:
                             0 - Copy default files.
                             1 - Run netconvert & polyconvert.
                             2 - Run ptlines2flows.py.
                             3 - Generate parking areas.
                             4 - Generate parking area rerouters.
                             5 - Generate taxi stands.
                             6 - Generate taxi stands rerouters.
                             7 - Extract TAZ from administrative boundaries.
                             8 - Generate OD-matrix.
                             9 - Generate SUMOActivityGen defaults.
                            10 - Run SUMOActivityGen.
                            11 - Launch SUMO.
                            12 - Report.
  --profiling           Enable Python3 cProfile feature.
  --no-profiling        Disable Python3 cProfile feature.
  --gui                 Enable SUMO GUI
  --no-gui              Disable SUMO GUI
  --local-defaults      Uses the default folder and files defined locally. If
                        not enabled, uses the files contained in the
                        sumo/tools/contributed/saga folder.
```

## The SUMOActivityGen mobility generator

Documentation in [docs/SUMOActivityGen.md](SUMOActivityGen.md).

`python3 activitygen.py -c configuration.json`
