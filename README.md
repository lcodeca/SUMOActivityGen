# SUMOActivityGen
Activity-based Mobility Generation for SUMO Simulator

Contacts: Lara CODECA [lara.codeca@gmail.com], Jerome HAERRI [haerri@eurecom.fr]

This program and the accompanying materials are made available under the
terms of the Eclipse Public License 2.0 which is available at http://www.eclipse.org/legal/epl-2.0.

### Overview
This project is in its early stages, and it is still under development.
* It can be tested and explored using the configuration files provided by [MoSTScenario](https://github.com/lcodeca/MoSTScenario) and `bash most.generator.sh`
* The complete generation of a scenario from OSM can be done using `python3 scenarioFromOSM.py --osm {osm file} --out {putput directory}`. All the generated files are going to be in the output directory.

![SUMOActivityGen Overview](https://github.com/lcodeca/SUMOActivityGen/blob/master/SUMOActivityGen.png)

![Scenario Generation Overview](https://github.com/lcodeca/SUMOActivityGen/blob/master/ScenarioGenerator.png)

### Due to some changes in the SUMO development version of the TraCI APIs, the master branch is not compatible with SUMO 1.2.0
* _Release v0.1 is compatible with SUMO 1.2.0_
