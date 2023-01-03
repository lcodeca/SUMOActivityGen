#!/usr/bin/env python3

""" SUMO Activity-Based Mobility Generator

    Author: Lara CODECA

    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

import argparse
import collections
import cProfile

import io
import json
import logging
import os
from pprint import pformat
import pstats
import sys
import xml.etree.ElementTree

from enum import Enum

import numpy
from numpy.random import RandomState
from tqdm import tqdm

from agsrc import activities, environment, sagaexceptions, sumoutils

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
    import traci
    import traci.constants as tc
    from traci._simulation import Stage
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")


def get_options(cmd_args):
    """Argument Parser."""
    parser = argparse.ArgumentParser(
        prog="activitygen.py",
        usage="%(prog)s -c configuration.json",
        description="SUMO Activity-Based Mobility Generator",
    )
    parser.add_argument(
        "-c", type=str, dest="config", required=True, help="JSON configuration file."
    )
    parser.add_argument(
        "--profiling",
        dest="profiling",
        action="store_true",
        help="Enable Python3 cProfile feature.",
    )
    parser.add_argument(
        "--no-profiling",
        dest="profiling",
        action="store_false",
        help="Disable Python3 cProfile feature.",
    )
    parser.set_defaults(profiling=False)
    return parser.parse_args(cmd_args)


class MobilityGenerator:
    """Generates intermodal mobility for SUMO starting from a synthetic population."""

    class ModeShare(Enum):
        """
        Selector between two interpretation of the values used for the modes:
            - PROBABILITY: only one mode is selected using the given probability.
            - WEIGHT: all the modes are generated, the cost is multiplied by the given weight and
                        only the cheapest solution is used.
        """

        PROBABILITY = 1
        WEIGHT = 2

    LAST_STOP_PLACEHOLDER = -42.42

    def _configure_loggers(self):
        """Setup the console and file logger."""
        self.logger = logging.getLogger("ActivityGen")
        self.logger.setLevel(logging.DEBUG)
        _console_handler = logging.StreamHandler()
        _console_handler.setLevel(logging.INFO)
        _console_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
        )
        self.logger.addHandler(_console_handler)
        _file_handler = logging.FileHandler(
            "{}.debug.log".format(self._conf["outputPrefix"])
        )
        _file_handler.setLevel(logging.DEBUG)
        _file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s]:[%(name)s]:[%(module)s:%(funcName)s:%(lineno)d]:[%(levelname)s] "
                "%(message)s"
            )
        )
        self.logger.addHandler(_file_handler)

    def __init__(self, conf, profiling=False):
        """
        Initialize the synthetic population.
            :param conf: distionary with the configurations
            :param profiling=False: enable cProfile
        """
        self._conf = conf

        # Setting the LOGGING
        self._configure_loggers()

        if not "modeSelection" in conf["intermodalOptions"]:
            raise Exception(
                'The parameter "modeSelection" in "intermodalOptions" must be defined.'
            )
        self._mode_interpr = None
        if conf["intermodalOptions"]["modeSelection"] == "PROBABILITY":
            self._mode_interpr = self.ModeShare.PROBABILITY
        elif conf["intermodalOptions"]["modeSelection"] == "WEIGHT":
            self._mode_interpr = self.ModeShare.WEIGHT
        else:
            raise Exception(
                'The parameter "modeSelection" in "intermodalOptions" must be set to '
                '"PROBABILITY" or "WEIGHT".'
            )

        self._max_retry_number = 1000
        if "maxNumTry" in conf:
            self._max_retry_number = conf["maxNumTry"]

        self._profiling = profiling

        self._random_generator = RandomState(seed=self._conf["seed"])

        self.logger.info("Starting TraCI with file %s.", conf["sumocfg"])
        traci.start(["sumo", "-c", conf["sumocfg"]], traceFile="traci.log")

        if not "vTypes" in self._conf["SUMOadditionals"]:
            raise Exception(
                'The parameter "vTypes" in "SUMOadditionals" must be defined.'
            )
        self._vtype_to_vclasses = collections.defaultdict(list)
        self._load_vtypes_distributions(self._conf["SUMOadditionals"]["vTypes"])

        self._env = environment.Environment(
            conf, sumo=traci, profiling=profiling, logger=self.logger
        )
        self._chains = activities.Activities(
            conf,
            sumo=traci,
            environment=self._env,
            profiling=profiling,
            logger=self.logger,
        )

        self._all_trips = collections.defaultdict(dict)

        self.logger.info("Computing the number of entities for each mobility slice..")
        self._compute_entities_per_slice()

    def generate(self):
        """Generates and saves the mobility."""
        self._mobility_generation()
        self._save_mobility()
        self._close_traci()

    def _mobility_generation(self):
        """Generate the mobility for the synthetic population."""
        self.logger.info("Generating on-deman fleet..")
        self._generate_taxi_fleet()
        self.logger.info("Generating trips for each mobility slice..")
        self._compute_trips_per_slice()

    def _save_mobility(self):
        """Save the generated trips to files."""
        self.logger.info("Saving trips files..")
        if self._conf["mergeRoutesFiles"]:
            self._saving_trips_to_single_file()
        else:
            self._saving_trips_to_files()

    def _close_traci(self):
        """Artefact to close TraCI properly."""
        self.logger.debug("Closing TraCI.")
        traci.close()

    ## ---------------------------------------------------------------------------------------- ##
    ##                                Loaders                                                   ##
    ## ---------------------------------------------------------------------------------------- ##

    def _load_vtypes_distributions(self, filename):
        """Load vTypeDistributions from XML file."""
        xml_tree = xml.etree.ElementTree.parse(filename).getroot()
        for child in xml_tree:
            if child.tag == "vType":
                self._vtype_to_vclasses[child.attrib["id"]].append(
                    child.attrib["vClass"]
                )
            elif child.tag == "vTypeDistribution":
                for subchild in child:
                    self._vtype_to_vclasses[child.attrib["id"]].append(
                        subchild.attrib["vClass"]
                    )
        self.logger.debug("vTypes loaded: \n%s", pformat(self._vtype_to_vclasses))

    ## ---------------------------------------------------------------------------------------- ##
    ##                                Mobility Generation                                       ##
    ## ---------------------------------------------------------------------------------------- ##

    ## ----                       On-demand fleet generation                               ---- ##

    ON_DEMAND_STR = """
    <vehicle id="{id}" type="on-demand" depart="0.0">
        <route edges="{edge}"/>
        <stop lane="{lane}" startPos="1.0" endPos="-1.0" triggered="person"/>
    </vehicle>"""

    def _generate_taxi_fleet(self):
        """Generate the number of on-demand vehicles set in the configuration file."""
        self.logger.info(
            "On-demand fleet expected size of %d",
            self._conf["intermodalOptions"]["taxiFleetSize"],
        )
        _fleet = []
        for _vehicle_id in tqdm(
            range(self._conf["intermodalOptions"]["taxiFleetSize"])
        ):
            _name = "on-demand.{}".format(_vehicle_id)
            _lane = self._env.get_random_lane_from_tazs()
            if _lane is None:
                continue
            _edge = _lane.split("_")[0]
            _fleet.append(
                {
                    "id": _name,
                    "depart": 0.0,
                    "string": self.ON_DEMAND_STR.format(
                        id=_name, edge=_edge, lane=_lane
                    ),
                }
            )
        self.logger.info("Generated an on-demant fleet of %d vehicles.", len(_fleet))
        self._all_trips["on-demand-fleet"][0] = _fleet
        with open(
            "{}.trips.dump.json".format(self._conf["outputPrefix"]), "w"
        ) as openfile:
            json.dump(
                {
                    "name": "on-demand-fleet",
                    "depart": 0,
                    "taxi": _fleet,
                },
                openfile,
                indent=2,
            )

    ## ----                           Person trips generation                               ---- ##

    def _compute_entities_per_slice(self):
        """
        Compute the absolute number of entities that are going to be created
        for each moblitiy slice, given a population.
        """
        self.logger.info("Population: %d", self._conf["population"]["entities"])

        for m_slice in self._conf["slices"].keys():
            self._conf["slices"][m_slice]["tot"] = int(
                self._conf["population"]["entities"]
                * self._conf["slices"][m_slice]["perc"]
            )
            self.logger.info("\t %s: %d", m_slice, self._conf["slices"][m_slice]["tot"])

    def _compute_trips_per_slice(self):
        """Compute the trips for the synthetic population for each mobility slice."""

        total = 0

        _modes_stats = collections.defaultdict(int)
        _chains_stats = collections.defaultdict(int)

        for name, m_slice in self._conf["slices"].items():
            self.logger.info(
                "[%s] Computing %d trips from %s to %s ... ",
                name,
                m_slice["tot"],
                m_slice["loc_origin"],
                m_slice["loc_primary"],
            )

            ## Activity chains preparation
            activity_chains = []
            activity_chains_weights = []
            for _weight, _chain, _modes in m_slice["activityChains"]:
                activity_chains.append((_chain, _modes))
                activity_chains_weights.append(_weight)
            activity_index = range(len(activity_chains))

            if self._profiling:
                _pr = cProfile.Profile()
                _pr.enable()

            for entity_id in tqdm(range(m_slice["tot"])):
                ## Select the activity chain
                _index = self._random_generator.choice(
                    activity_index, p=activity_chains_weights
                )
                _chain, _modes = activity_chains[_index]
                self.logger.debug(
                    "_compute_trips_per_slice: Chain: %s", "{}".format(_chain)
                )
                self.logger.debug(
                    "_compute_trips_per_slice: Modes: %s", "{}".format(_modes)
                )

                _person_trip = None

                # (Intermodal) trip
                _final_chain = None
                _stages = None
                _error_counter = 0
                while not _person_trip and _error_counter < self._max_retry_number:
                    try:
                        self.logger.debug(
                            " ====================== _generate_trip ====================== "
                        )
                        _final_chain, _stages, _selected_mode = self._generate_trip(
                            self._conf["taz"][m_slice["loc_origin"]],
                            self._conf["taz"][m_slice["loc_primary"]],
                            _chain,
                            _modes,
                        )

                        ## Generating departure time
                        _depart = numpy.round(_final_chain[1].start, decimals=2)
                        if _depart < 0.0:
                            _msg_error = (
                                "Negative departure time generated: {}.".format(_depart)
                            )
                            _msg_error += (
                                " Required fixing of the start time of the initial"
                            )
                            _msg_error += " activities. "
                            # raise sagaexceptions.TripGenerationGenericError(_msg_error)
                            _depart = 0.0
                            _msg_error += " Departure set to 0.0."
                            self.logger.critical(_msg_error)

                        if _depart not in self._all_trips[name].keys():
                            self._all_trips[name][_depart] = []

                        _person_trip = {
                            "id": "{}_{}".format(name, entity_id),
                            "depart": _depart,
                            "stages": _stages,
                        }

                        complete_trip = self._generate_sumo_trip_from_activitygen(
                            _person_trip
                        )

                        _person_trip["string"] = complete_trip
                        ## For statistical purposes.
                        _modes_stats[_selected_mode] += 1
                        _chains_stats[self._hash_final_chain(_final_chain)] += 1

                    except sagaexceptions.TripGenerationGenericError:
                        _person_trip = None
                        _error_counter += 1
                    finally:
                        self.logger.debug(
                            " ====================== DONE ====================== "
                        )

                if _person_trip:
                    # Trip creation
                    self._all_trips[name][_depart].append(_person_trip)
                    with open(
                        "{}.trips.dump.json".format(self._conf["outputPrefix"]), "a"
                    ) as openfile:
                        json.dump(
                            {
                                "name": name,
                                "depart": _depart,
                                "personTrip": pformat(_person_trip),
                            },
                            openfile,
                            indent=2,
                        )
                    self.logger.debug("Generated: %s", _person_trip["string"])
                    total += 1

                else:
                    self.logger.critical(
                        "_generate_trip from %s to %s generated %d errors, "
                        "trip generation aborted..",
                        self._conf["taz"][m_slice["loc_origin"]],
                        self._conf["taz"][m_slice["loc_primary"]],
                        _error_counter,
                    )

            if self._profiling:
                _pr.disable()
                _s = io.StringIO()
                _ps = pstats.Stats(_pr, stream=_s).sort_stats("cumulative")
                _ps.print_stats(10)
                print(_s.getvalue())
                input("Press any key to continue..")

        self.logger.info("Generated %d trips.", total)
        self.logger.info("Mode splits:")
        for mode, value in _modes_stats.items():
            self.logger.info("\t %s: %d (%.2f).", mode, value, float(value / total))
        self.logger.info("Activity chains splits:")
        for chain, value in _chains_stats.items():
            self.logger.info("\t %s: %d (%.2f).", chain, value, float(value / total))

    @staticmethod
    def _hash_final_chain(chain):
        activity_list = list()
        for pos in range(1, len(chain) + 1):
            activity_list.append(chain[pos].activity)
        return pformat(activity_list)

    ## ---- Functions for _compute_trips_per_slice:
    #                                       _generate_trip, _generate_intermodal_trip_traci ---- ##

    def _generate_trip(self, from_area, to_area, activity_chain, modes):
        """Returns the trip for the given activity chain."""

        trip = None
        solutions = []

        ## Mode selection, weight or probability
        _interpr_modes = None
        if self._mode_interpr == self.ModeShare.PROBABILITY:
            _probs = []
            _vals = []
            for mode, prob in modes:
                _vals.append(mode)
                _probs.append(prob)
            selection = self._random_generator.choice(_vals, p=_probs)
            _interpr_modes = [[selection, 1.0]]  ## Unique mode, without weight.
        else:
            _interpr_modes = modes

        for mode, weight in _interpr_modes:

            ### DEBUGGING MODES: # <==================== !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # mode = 'walk'
            # mode = 'bicycle'
            # mode = 'public'
            # mode = 'passenger'
            # mode = 'motorcycle'
            # mode = 'on-demand'
            # mode = 'commercial'
            # mode = 'special'
            # mode = 'evehicle'

            _person_steps = None
            _error_counter = 0
            while not _person_steps and _error_counter < self._max_retry_number:
                try:
                    (
                        _person_steps,
                        _person_stages,
                    ) = self._generate_intermodal_trip_traci(
                        from_area, to_area, activity_chain, mode
                    )
                except sagaexceptions.TripGenerationGenericError:
                    _person_steps = None
                    _error_counter += 1

            if _person_steps:
                ## Cost computation.
                solutions.append(
                    (
                        sumoutils.cost_from_route(_person_steps) * weight,
                        _person_steps,
                        _person_stages,
                        mode,
                    )
                )
            else:
                self.logger.critical(
                    '_generate_intermodal_trip_traci from "%s" to "%s" with "%s" '
                    "generated %d errors, trip generation aborted..",
                    from_area,
                    to_area,
                    mode,
                    _error_counter,
                )

        ## Compose the final person trip.
        if solutions:
            ## For the moment, the best solution is the one with minor cost.
            best = sorted(solutions)[0]  ## Ascending.
            trip = (best[2], best[1], best[3])  ## _person_stages, _person_steps, mode
        else:
            raise sagaexceptions.TripGenerationRouteError(
                "No solution foud for chain {} and modes {}.".format(
                    activity_chain, _interpr_modes
                )
            )
        return trip

    def _generate_intermodal_trip_traci(self, from_area, to_area, activity_chain, mode):
        """Return the person trip for a given mode generated with TraCI"""

        self.logger.debug(
            " ====================== _generate_intermodal_trip_traci ====================== "
        )

        _person_stages = self._chains.generate_person_stages(
            from_area, to_area, activity_chain, mode
        )

        _person_steps = []
        _current_depart_time = None

        _mode, _ptype, _vtype = sumoutils.get_intermodal_mode_parameters(
            mode, self._conf["intermodalOptions"]["vehicleAllowedParking"]
        )
        self.logger.debug("Mode: %s ==> %s, %s, %s", mode, _mode, _ptype, _vtype)

        for pos in range(1, len(_person_stages) + 1):
            stage = _person_stages[pos]
            self.logger.debug("STAGE %d: %s", pos, pformat(stage))
            # findIntermodalRoute(self, fromEdge, toEdge, modes='', depart=-1.0,
            #                     routingMode=0, speed=-1.0, walkFactor=-1.0,
            #                     departPos=0.0, arrivalPos=-1073741824, departPosLat=0.0,
            #                     pType='', vType='', destStop='')
            if _current_depart_time is None:
                _current_depart_time = stage.start

            route = None

            if (
                stage.activity != "Home"
                and _vtype in self._conf["intermodalOptions"]["vehicleAllowedParking"]
            ):
                # TRIP WITH PARKING REQUIREMENTS
                #  If the vtype is among the one that require parking, and we are not going home,
                #  look for a parking and build the additional walk back and forth.
                p_id, p_edge, _last_mile = self._env.find_closest_parking(stage.toEdge)
                if _last_mile:
                    route = traci.simulation.findIntermodalRoute(
                        stage.fromEdge,
                        p_edge,
                        depart=_current_depart_time,
                        walkFactor=0.9,
                        modes=_mode,
                        pType=_ptype,
                        vType=_vtype,
                    )
                    if route and not isinstance(route, list):
                        # list in until SUMO 1.4.0 included, tuple onward
                        route = list(route)
                    if (
                        sumoutils.is_valid_route(
                            mode,
                            route,
                            self._conf["intermodalOptions"]["vehicleAllowedParking"],
                        )
                        and route[-1].type == tc.STAGE_DRIVING
                    ):
                        route[-1].destStop = p_id
                        route[-1].arrivalPos = self._env.get_parking_position(p_id)
                        route.extend(_last_mile)
                    else:
                        route = None
                if route:
                    ## build the waiting to destination (if required)
                    if stage.duration:
                        route.append(self._generate_waiting_stage(stage))

                    ## build the walk back to the parking
                    walk_back = traci.simulation.findIntermodalRoute(
                        stage.toEdge, p_edge, walkFactor=0.9, pType="pedestrian"
                    )
                    if route and not isinstance(route, list):
                        # list in until SUMO 1.4.0 included, tuple onward
                        route = list(route)
                    walk_back[-1].arrivalPos = self._env.get_parking_position(p_id)
                    route.extend(walk_back)

                    ## update the next stage to make it start from the parking
                    if pos + 1 in _person_stages:
                        _person_stages[pos + 1] = _person_stages[pos + 1]._replace(
                            fromEdge=p_edge
                        )

            elif _mode in ["public", "taxi"] or _ptype in ["pedestrian"]:
                # TRIP WITHOUT PARKING REQUIREMENTS, and no walk back and forth to mode change.
                #  If the _mode is among the one that are not generating the walk back and forth,
                #  the route provided by findIntermodalRoute.
                route = traci.simulation.findIntermodalRoute(
                    stage.fromEdge,
                    stage.toEdge,
                    arrivalPos=stage.arrivalPos,
                    depart=_current_depart_time,
                    walkFactor=0.9,
                    modes=_mode,
                    pType=_ptype,
                    vType=_vtype,
                )
                if not sumoutils.is_valid_route(
                    mode,
                    route,
                    self._conf["intermodalOptions"]["vehicleAllowedParking"],
                ):
                    route = None
                if route and not isinstance(route, list):
                    # list in until SUMO 1.4.0 included, tuple onward
                    route = list(route)
                if route:
                    ## Set the arrival position in the edge
                    route[-1].arrivalPos = stage.arrivalPos
                    ## Add stop
                    if stage.duration:
                        route.append(self._generate_waiting_stage(stage))
            else:
                # TRIP WITHOUT PARKING REQUIREMENTS, BUT with walk back and forth to mode change.
                #  If the _mode  requires a mode change, the walk back and forth needs to be
                #  added after the waiting at destination due to the activity.
                route = traci.simulation.findIntermodalRoute(
                    stage.fromEdge,
                    stage.toEdge,
                    arrivalPos=stage.arrivalPos,
                    depart=_current_depart_time,
                    walkFactor=0.9,
                    modes=_mode,
                    pType=_ptype,
                    vType=_vtype,
                )
                if not sumoutils.is_valid_route(
                    mode,
                    route,
                    self._conf["intermodalOptions"]["vehicleAllowedParking"],
                ):
                    route = None
                if route and not isinstance(route, list):
                    # list in until SUMO 1.4.0 included, tuple onward
                    route = list(route)
                if route:
                    ## Add stop
                    if stage.duration:
                        route.append(self._generate_waiting_stage(stage))
                    if not stage.final:
                        # If it's not the final stage,
                        #   check if it's required to add the walk back to the vehicle and
                        #   change the stage.fromEdge of the next stage.
                        self.logger.debug(
                            "Check walking back requirements for route \n%s",
                            pformat(route),
                        )
                        if route[-2].type == tc.STAGE_WALKING:
                            # generate the walk back
                            from_edge = route[-2].edges[-1]  # end ow the walking stage
                            to_edge = route[-3].edges[-1]  # end of the driving stage
                            arrival_position = self._env.get_arrival_pos_from_edge(
                                route[-3].edges[-1], route[-3].arrivalPos
                            )  # parked vehicle
                            if arrival_position is None:
                                raise sagaexceptions.TripGenerationRouteError(
                                    "Arrival edge for the the walk back is too short."
                                )
                            self.logger.debug(
                                "Walk back from %s to %s (%.2f --> %.2f)",
                                from_edge,
                                to_edge,
                                route[-3].arrivalPos,
                                arrival_position,
                            )
                            walk_back = traci.simulation.findIntermodalRoute(
                                from_edge,
                                to_edge,
                                arrivalPos=arrival_position,
                                depart=(_current_depart_time + route[-1].travelTime),
                                walkFactor=0.9,
                                pType="pedestrian",
                            )
                            if not walk_back:
                                raise sagaexceptions.TripGenerationRouteError(
                                    "Route not found for the walk back from {} to {}.".format(
                                        from_edge, to_edge
                                    )
                                )
                            self.logger.debug("Walk back: \n%s", pformat(walk_back))
                            if walk_back and not isinstance(walk_back, list):
                                # list in until SUMO 1.4.0 included, tuple onward
                                walk_back = list(walk_back)
                            route.extend(walk_back)
                            _person_stages[pos + 1] = _person_stages[pos + 1]._replace(
                                fromEdge=route[-1].edges[-1]
                            )

            ## Finalize the Journey
            if route and stage.final:
                ## fix the last stop with 1.0 duration
                if route[-1].type == tc.STAGE_WAITING:
                    route[-1].travelTime = 1.0
                    route[-1].cost = 1.0
                else:
                    raise sagaexceptions.TripGenerationInconsistencyError(
                        "Final route misses last waiting stage.", _person_stages
                    )
                ## change the last triggered ride with LAST_STOP_PLACEHOLDER to fix the last stop
                if _mode not in ["public", "taxi"] and _ptype not in ["pedestrian"]:
                    _pos = len(route) - 1
                    while _pos >= 0:
                        if route[_pos].type == tc.STAGE_DRIVING:
                            route[_pos].destStop = self.LAST_STOP_PLACEHOLDER
                            break
                        _pos -= 1

            self.logger.debug("Route: \n%s", pformat(route))

            if route is None:
                raise sagaexceptions.TripGenerationRouteError(
                    "Route not found between {} and {}.".format(
                        stage.fromEdge, stage.toEdge
                    )
                )

            ## Add the stage to the full planned trip.
            for step in route:
                _current_depart_time += step.travelTime
                _person_steps.append(step)

            self.logger.debug(
                "====================== Stage done. ======================"
            )
        return _person_steps, _person_stages

    def _generate_waiting_stage(self, stage):
        """Builds a STAGE_WAITING type of stage compatible with findIntermodalRoute."""
        wait = Stage(
            type=tc.STAGE_WAITING,
            description=stage.activity,
            edges="{}_0".format(stage.toEdge),
            travelTime=stage.duration,
            cost=stage.duration,
            vType=None,
            line=None,
            destStop=None,
            length=None,
            intended=None,
            depart=None,
            departPos=None,
            arrivalPos=stage.arrivalPos,
        )
        self.logger.debug("WAITING Stage: %s", pformat(wait))
        return wait

    ## ---------------------------------------------------------------------------------------- ##
    ##                                       TraCI to XML                                       ##
    ## ---------------------------------------------------------------------------------------- ##

    ROUTES_TPL = """<?xml version="1.0" encoding="UTF-8"?>

<!-- Generated with SUMO Activity-Based Mobility Generator [https://github.com/lcodeca/SUMOActivityGen] -->

<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd"> {trips}
</routes>"""

    VEHICLE = """
    <vehicle id="{id}" type="{v_type}" depart="{depart}" departLane="best" arrivalPos="{arrival}">{route}{stop}
    </vehicle>"""

    ROUTE = """
        <route edges="{edges}"/>"""

    STOP_PARKING_TRIGGERED = """
        <stop parkingArea="{id}" triggered="true" expected="{person}"/>"""

    STOP_EDGE_TRIGGERED = """
        <stop lane="{lane}" parking="true" startPos="{start}" endPos="{end}" triggered="true" expected="{person}"/>"""  # pylint: disable=c0301

    ONDEMAND_TRIGGERED = """
        <stop lane="{lane}" parking="true" startPos="{start}" endPos="{end}" duration="1.0"/>"""

    FINAL_STOP = """
        <stop lane="{lane}" duration="1.0"/>"""

    PERSON = """
    <person id="{id}" type="pedestrian" depart="{depart}">{stages}
    </person>"""

    WAIT = """
        <stop lane="{lane}" duration="{duration}" actType="{action}"/>"""

    WALK = """
        <walk edges="{edges}"/>"""

    WALK_W_ARRIVAL = """
        <walk edges="{edges}" arrivalPos="{arrival}"/>"""

    WALK_BUS = """
        <walk edges="{edges}" busStop="{busStop}"/>"""

    RIDE_BUS = """
        <ride busStop="{busStop}" lines="{lines}" intended="{intended}" depart="{depart}"/>"""

    RIDE_TRIGGERED = """
        <ride from="{from_edge}" to="{to_edge}" arrivalPos="{arrival}" lines="{vehicle_id}"/>"""

    VEHICLE_TRIGGERED = """
    <vehicle id="{id}" type="{v_type}" depart="triggered">{route}{stops}
    </vehicle>"""

    VEHICLE_TRIGGERED_DEPART = """
    <vehicle id="{id}" type="{v_type}" depart="triggered" departPos="{depart}">{route}{stops}
    </vehicle>"""

    def _generate_stop_position(self, edge, position):
        """ """
        start = position - self._conf["stopBufferDistance"] / 2.0
        end = position + self._conf["stopBufferDistance"] / 2.0
        if start <= 0.0:
            start = 1.0
            end = self._conf["stopBufferDistance"]
        if position > end or end >= self._env.sumo_network.getEdge(edge).getLength():
            end = -1.0
        return start, end

    def _generate_sumo_trip_from_activitygen(self, person):
        """Generate the XML string for SUMO route file from a person-trip."""
        self.logger.debug(
            " ............... _generate_sumo_trip_from_activitygen ............... "
        )
        self.logger.debug("\n%s", pformat(person))
        self.logger.debug(" ............... computation ............... ")
        complete_trip = ""
        triggered = ""
        _tr_id = "{}_tr".format(person["id"])
        _triggered_vtype = ""
        _triggered_route = []
        _triggered_stops = ""
        stages = ""
        _last_arrival_pos = None
        _internal_consistency_check = []
        _waiting_stages = []
        for stage in person["stages"]:
            self.logger.debug("Stage \n%s", pformat(stage))
            if stage.type == tc.STAGE_WAITING:
                _waiting_stages.append(stage)
                stages += self.WAIT.format(
                    lane=stage.edges,
                    duration=stage.travelTime,
                    action=stage.description,
                )
            elif stage.type == tc.STAGE_WALKING:
                if stage.destStop:
                    stages += self.WALK_BUS.format(
                        edges=" ".join(stage.edges), busStop=stage.destStop
                    )
                else:
                    if stage.arrivalPos:
                        stages += self.WALK_W_ARRIVAL.format(
                            edges=" ".join(stage.edges), arrival=stage.arrivalPos
                        )
                        _last_arrival_pos = stage.arrivalPos
                    else:
                        stages += self.WALK.format(edges=" ".join(stage.edges))
            elif stage.type == tc.STAGE_DRIVING:
                if stage.line != stage.intended:
                    # Public Transports
                    # !!! vType MISSING !!! line=164:0, intended=pt_bus_164:0.50
                    # intended is the transport id, so it must be different
                    stages += self.RIDE_BUS.format(
                        busStop=stage.destStop,
                        lines=stage.line,
                        intended=stage.intended,
                        depart=stage.depart,
                    )
                else:
                    # Triggered vehicle (vTyep == line == intended)
                    # vType=bicycle, line=bicycle, intended=bicycle
                    # vType=passenger, line=passenger, intended=passenger
                    # vType=motorcycle, line=motorcycle, intended=motorcycle
                    # vType=on-demand, line=on-demand, intended=on-demand
                    _ride_id = None
                    if stage.intended == "on-demand":
                        _ride_id = "taxi"
                    else:
                        ## add to the existing one
                        _ride_id = _tr_id
                        if _triggered_route:
                            ## check for contiguity
                            if _triggered_route[-1] != stage.edges[0]:
                                _error_message = (
                                    "Triggered vehicle {} has a broken route.".format(
                                        stage.intended
                                    )
                                )
                                self.logger.warning(_error_message)
                                raise sagaexceptions.TripGenerationInconsistencyError(
                                    _error_message, pformat(person["stages"])
                                )
                            ## remove the duplicated edge
                            _triggered_route.extend(stage.edges[1:])
                        else:
                            ## nothing to be "fixed"
                            _triggered_route.extend(stage.edges)
                        self.logger.debug("_triggered_route: %s", _triggered_route)
                        _triggered_vtype = stage.vType
                        _stop = ""
                        self.logger.debug(
                            "travelTime: %f - destStop: %s",
                            stage.travelTime,
                            stage.destStop,
                        )
                        if stage.destStop == self.LAST_STOP_PLACEHOLDER:
                            self.logger.debug("Final stop reached.")
                            _stop = self.FINAL_STOP.format(
                                lane=self._env.get_stopping_lane(
                                    stage.edges[-1],
                                    self._vtype_to_vclasses[_triggered_vtype],
                                )
                            )
                        else:
                            if (
                                stage.destStop
                                and stage.vType
                                in self._conf["intermodalOptions"][
                                    "vehicleAllowedParking"
                                ]
                            ):
                                self.logger.debug(
                                    "Generate the stop in a parking area."
                                )
                                _stop = self.STOP_PARKING_TRIGGERED.format(
                                    id=stage.destStop, person=person["id"]
                                )
                            else:
                                self.logger.debug(
                                    "Generate the stop on the side of the road."
                                )
                                start, end = self._generate_stop_position(
                                    stage.edges[-1], stage.arrivalPos
                                )
                                self.logger.debug("Stop from %f to %f", start, end)
                                _stop = self.STOP_EDGE_TRIGGERED.format(
                                    lane=self._env.get_stopping_lane(
                                        stage.edges[-1],
                                        self._vtype_to_vclasses[_triggered_vtype],
                                    ),
                                    person=person["id"],
                                    start=start,
                                    end=end,
                                )
                                self.logger.debug("_stop: %s", _stop)
                        _triggered_stops += _stop

                        self.logger.debug("_triggered_stops: %s", _triggered_stops)

                    stages += self.RIDE_TRIGGERED.format(
                        from_edge=stage.edges[0],
                        to_edge=stage.edges[-1],
                        vehicle_id=_ride_id,
                        arrival=stage.arrivalPos,
                    )

                    self.logger.debug("Stages: %s", stages)

            self.logger.debug(" -- Next stage -- ")

        ## fixing the personal triggered vehicles
        if _triggered_route:
            _route = self.ROUTE.format(edges=" ".join(_triggered_route))
            triggered += self.VEHICLE_TRIGGERED.format(
                id=_tr_id,
                v_type=_triggered_vtype,
                route=_route,
                stops=_triggered_stops,
                arrival="random",
                depart="random",
            )

        ## waiting stages consistency test
        if not _waiting_stages:
            self.logger.warning("Person plan does not have any waiting stages.")
            raise sagaexceptions.TripGenerationInconsistencyError(
                "Person plan does not have any waiting stages.", person["stages"]
            )

        ## result
        complete_trip += triggered
        complete_trip += self.PERSON.format(
            id=person["id"], depart=person["depart"], stages=stages
        )

        self.logger.debug("Complete trip: \n%s", complete_trip)
        return complete_trip

    ## ---------------------------------------------------------------------------------------- ##
    ##                                Saving trips to files                                     ##
    ## ---------------------------------------------------------------------------------------- ##

    def _saving_trips_to_files(self):
        """Saving all the trips to files divided by slice."""
        for name, dict_trips in self._all_trips.items():
            filename = "{}{}.rou.xml".format(self._conf["outputPrefix"], name)
            with open(filename, "w", encoding="utf8") as tripfile:
                all_trips = ""
                for time in sorted(dict_trips.keys()):
                    for person in dict_trips[time]:
                        all_trips += person["string"]

                tripfile.write(self.ROUTES_TPL.format(trips=all_trips))
            self.logger.info("Saved %s", filename)

    def _saving_trips_to_single_file(self):
        """Saving all the trips into a single file."""
        ## Sort (by time) all the slice into one
        merged_trips = collections.defaultdict(list)
        for dict_trips in self._all_trips.values():
            for time in sorted(dict_trips.keys()):
                for person in dict_trips[time]:
                    merged_trips[time].append(person["string"])

        filename = "{}.merged.rou.xml".format(self._conf["outputPrefix"])
        with open(filename, "w", encoding="utf8") as tripfile:
            all_trips = ""
            for time in sorted(merged_trips.keys()):
                for person in merged_trips[time]:
                    all_trips += person

            tripfile.write(self.ROUTES_TPL.format(trips=all_trips))
            self.logger.info("Saved %s", filename)


def main(cmd_args):
    """Person Trip Activity-based Mobility Generation with PoIs and TAZ."""
    args = get_options(cmd_args)

    ## ========================              PROFILER              ======================== ##
    if args.profiling:
        profiler = cProfile.Profile()
        profiler.enable()
    ## ========================              PROFILER              ======================== ##

    print("Loading configuration file {}.".format(args.config))
    MobilityGenerator(
        json.loads(open(args.config).read()), profiling=args.profiling
    ).generate()

    ## ========================              PROFILER              ======================== ##
    if args.profiling:
        profiler.disable()
        results = io.StringIO()
        pstats.Stats(profiler, stream=results).sort_stats("cumulative").print_stats(25)
        print(results.getvalue())
    ## ========================              PROFILER              ======================== ##

    print("Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
