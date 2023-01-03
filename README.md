# SUMOActivityGen

Activity-based Mobility Generation for SUMO Simulator

Contacts: Lara CODECA [lara.codeca@gmail.com]

Cite: [_L. CODECA, J. ERDMANN, V. CAHILL, J. HAERRI, "SAGA: An Activity-based Multi-modal Mobility Scenario Generator for SUMO". SUMO User Conference 2020 - From Traffic Flow to Mobility Modeling._](https://www.researchgate.net/publication/346485853_SAGA_An_Activity-based_Multi-modal_Mobility_Scenario_Generator_for_SUMO)
The [presentation](https://www.youtube.com/watch?v=b-ZvQ0XbVvM) is available on YouTube.

This program and the accompanying materials are made available under the
terms of the Eclipse Public License 2.0 which is available at <http://www.eclipse.org/legal/epl-2.0>.

This Source Code may also be made available under the following Secondary Licenses when the conditions for such availability set forth in the Eclipse Public License, v. 2.0 are satisfied: GNU General Public License version 3 <https://www.gnu.org/licenses/gpl-3.0.txt>.

## Status

Weekly update against the SUMO master branch.

[![linux-master](https://github.com/lcodeca/SUMOActivityGen/actions/workflows/linux-master.yml/badge.svg)](https://github.com/lcodeca/SUMOActivityGen/actions/workflows/linux-master.yml) [![linux-release](https://github.com/lcodeca/SUMOActivityGen/actions/workflows/linux-release.yml/badge.svg)](https://github.com/lcodeca/SUMOActivityGen/actions/workflows/linux-release.yml) [![windows-pypi](https://github.com/lcodeca/SUMOActivityGen/actions/workflows/windows-pypi.yml/badge.svg)](https://github.com/lcodeca/SUMOActivityGen/actions/workflows/windows-pypi.yml)

## Overview

* The complete generation of a scenario from OSM can be done using `python3 scenarioFromOSM.py --osm {osm file} --out {output directory}`. All the generated files are going to be in the output directory.
* Alternatively, it can be tested and explored using the configuration files provided by [MoSTScenario](https://github.com/lcodeca/MoSTScenario) and in the `example` directory, starting from `bash most.generator.sh`.

The documentation is availalbe in the `docs` folder.

### Due to some changes in the SUMO development version of the TraCI APIs, the master branch may not compatible with SUMO 1.4.0

* _Release v0.2 is compatible with SUMO 1.4.0_
* _Release v0.1 is compatible with SUMO 1.2.0_

## Linting
For linting, use the `linting.sh` script. 