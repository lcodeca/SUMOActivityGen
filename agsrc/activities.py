#!/usr/bin/env python3

""" SUMO Activity-Based Mobility Generator - Activity console_handlerains

    Author: Lara CODECA

    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 whiconsole_handler is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

import collections
import logging
import os
from pprint import pformat
import sys

import numpy
from numpy.random import RandomState

from agsrc import sagaexceptions, sumoutils

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
    from traci.exceptions import TraCIException
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")


class Activities:
    """Generates the activity chains."""

    ## Activity
    Activity = collections.namedtuple(
        "Activity",
        ["activity", "fromEdge", "toEdge", "arrivalPos", "start", "duration", "final"],
    )
    Activity.__new__.__defaults__ = (None,) * len(Activity._fields)

    def __init__(self, conf, sumo, environment, logger, profiling=False):
        """
        Initialize the synthetic population.
            :param conf: distionary with the configurations
            :param sumo: already initialized SUMO simulation (TraCI or LibSUMO)
            :param profiling=False: enable cProfile
        """
        self._conf = conf
        self._sumo = sumo
        self._cache = {}
        self._env = environment
        self.logger = logger

        self._max_retry_number = 1000
        if "maxNumTry" in conf:
            self._max_retry_number = conf["maxNumTry"]

        self._profiling = profiling

        self._random_generator = RandomState(seed=self._conf["seed"])

    # Activity Locations

    def _stages_define_locations_position(self, person_stages):
        """Define the position of each location in the activity chain."""
        home_pos = None
        primary_pos = None

        for pos, stage in person_stages.items():
            if "Home" in stage.activity:
                if not home_pos:
                    home_pos = self._env.get_random_pos_from_edge(stage.toEdge)
                person_stages[pos] = stage._replace(arrivalPos=home_pos)
            elif "P-" in stage.activity:
                if not primary_pos:
                    primary_pos = self._env.get_random_pos_from_edge(stage.toEdge)
                person_stages[pos] = stage._replace(arrivalPos=primary_pos)
            else:
                ## Secondary activities
                person_stages[pos] = stage._replace(
                    arrivalPos=self._env.get_random_pos_from_edge(stage.toEdge)
                )

        return person_stages

    def _stages_define_main_locations(self, from_area, to_area, mode, estimated_start):
        """Define a generic Home and Primary activity location.
        The locations must be reachable in some ways.
        """
        ## Mode split:
        _mode, _ptype, _vtype = sumoutils.get_intermodal_mode_parameters(
            mode, self._conf["intermodalOptions"]["vehicleAllowedParking"]
        )

        route = None
        from_edge = None
        to_edge = None
        _retry_counter = 0
        while not route and _retry_counter < self._max_retry_number:
            _retry_counter += 1
            ## Origin and Destination Selection
            from_edge, to_edge = self._env.select_pair(from_area, to_area)
            from_allowed = (
                self._env.sumo_network.getEdge(from_edge).allows("pedestrian")
                and self._env.sumo_network.getEdge(from_edge).allows("passenger")
                and self._env.sumo_network.getEdge(from_edge).getLength()
                > self._conf["minEdgeAllowed"]
            )
            to_allowed = (
                self._env.sumo_network.getEdge(to_edge).allows("pedestrian")
                and self._env.sumo_network.getEdge(to_edge).allows("passenger")
                and self._env.sumo_network.getEdge(to_edge).getLength()
                > self._conf["minEdgeAllowed"]
            )
            if self._env.valid_pair(from_edge, to_edge) and from_allowed and to_allowed:
                try:
                    route = self._sumo.simulation.findIntermodalRoute(
                        from_edge,
                        to_edge,
                        depart=estimated_start,
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
                        self.logger.debug(
                            "_stages_define_main_locations: findIntermodalRoute mode unusable."
                        )
                except TraCIException:
                    self.logger.debug(
                        "_stages_define_main_locations: findIntermodalRoute FAILED."
                    )
                    route = None
            else:
                self.logger.debug(
                    "_stages_define_main_locations: unusable pair of edges."
                )
        if route:
            return from_edge, to_edge
        raise sagaexceptions.TripGenerationActivityError(
            "Locations for the main activities not found between {} and {} using {}.".format(
                from_area, to_area, mode
            )
        )

    def _stages_define_secondary_locations(self, person_stages, home, primary):
        """Define secondary activity locations."""
        for pos, stage in person_stages.items():
            if "S-" in stage.activity:
                ## look for what is coming before
                _prec = None
                _pos = pos - 1
                while not _prec and _pos in person_stages:
                    if "Home" in person_stages[_pos].activity:
                        _prec = "H"
                    elif "P-" in person_stages[_pos].activity:
                        _prec = "P"
                    _pos -= 1

                ## look for what is coming next
                _succ = None
                _pos = pos + 1
                while not _succ and _pos in person_stages:
                    if "Home" in person_stages[_pos].activity:
                        _succ = "H"
                    elif "P-" in person_stages[_pos].activity:
                        _succ = "P"
                    _pos += 1

                destination = None
                if _prec == "H" and _succ == "H":
                    destination = self._random_location_circle(
                        center=home, other=primary
                    )
                elif _prec == "P" and _succ == "P":
                    destination = self._random_location_circle(
                        center=primary, other=home
                    )
                elif _prec != _succ:
                    destination = self._random_location_ellipse(home, primary)
                else:
                    raise sagaexceptions.TripGenerationActivityError(
                        "Invalid sequence in the activity chain: {} --> {}".format(
                            _prec, _succ
                        ),
                        person_stages,
                    )

                person_stages[pos] = stage._replace(toEdge=destination)
        return person_stages

    def _random_location_circle(self, center, other):
        """Return a random edge in within a radius (*) from the given center.

        (*) Uses the ellipses defined by the foci center and other,
            and the major axe of 1.30 * distance between the foci.
        """
        try:
            length = self._get_cached_dist(center, other)
        except TraCIException:
            raise sagaexceptions.TripGenerationActivityError(
                "No route between {} and {}".format(center, other)
            )
        major_axe = length * 1.3
        minor_axe = numpy.sqrt(numpy.square(major_axe) - numpy.square(length))
        radius = minor_axe / 2.0

        self.logger.debug("_random_location_circle: %s [%.2f]", center, radius)
        edges = self._env.get_all_neigh_edges(center, radius)
        if not edges:
            raise sagaexceptions.TripGenerationActivityError(
                "No edges from {} with range {}.".format(center, length)
            )

        ret = self._random_generator.choice(edges)
        edges.remove(ret)
        allowed = (
            self._env.sumo_network.getEdge(ret).allows("pedestrian")
            and self._env.sumo_network.getEdge(ret).allows("passenger")
            and ret != center
            and ret != other
            and self._env.sumo_network.getEdge(ret).getLength()
            > self._conf["minEdgeAllowed"]
        )
        while edges and not allowed:
            ret = self._random_generator.choice(edges)
            edges.remove(ret)
            allowed = (
                self._env.sumo_network.getEdge(ret).allows("pedestrian")
                and self._env.sumo_network.getEdge(ret).allows("passenger")
                and ret != center
                and ret != other
                and self._env.sumo_network.getEdge(ret).getLength()
                > self._conf["minEdgeAllowed"]
            )

        if not edges:
            raise sagaexceptions.TripGenerationActivityError(
                "No valid edges from {} with range {}.".format(center, length)
            )
        return ret

    def _random_location_ellipse(self, focus1, focus2):
        """Return a random edge in within the ellipse defined by the foci,
        and the major axe of 1.30 * distance between the foci.
        """
        try:
            length = self._get_cached_dist(focus1, focus2)
            self.logger.debug(
                "_random_location_ellipse: %s --> %s [%.2f]", focus1, focus2, length
            )
        except TraCIException:
            raise sagaexceptions.TripGenerationActivityError(
                "No route between {} and {}".format(focus1, focus2)
            )

        major_axe = length * 1.3

        edges = self._env.get_all_neigh_edges(focus1, length)
        while edges:
            edge = self._random_generator.choice(edges)
            edges.remove(edge)
            if edge in (focus1, focus2):
                continue
            allowed = (
                self._env.sumo_network.getEdge(edge).allows("pedestrian")
                and self._env.sumo_network.getEdge(edge).allows("passenger")
                and self._env.sumo_network.getEdge(edge).getLength()
                > self._conf["minEdgeAllowed"]
            )
            if not allowed:
                continue
            try:
                first = self._get_cached_dist(focus1, edge)
                second = self._get_cached_dist(edge, focus2)
                if first + second <= major_axe:
                    self.logger.debug(
                        "_random_location_ellipse: %s --> %s [%.2f]",
                        focus1,
                        edge,
                        first,
                    )
                    self.logger.debug(
                        "_random_location_ellipse: %s --> %s [%.2f]",
                        edge,
                        focus2,
                        second,
                    )
                    return edge
            except TraCIException:
                pass

        raise sagaexceptions.TripGenerationActivityError(
            "No location available for _random_location_ellipse [{}, {}]".format(
                focus1, focus2
            )
        )

    # Chain

    def _get_estimated_activity_time_from_chain(self, activity_chain):
        """Returns the first usable time for any given chain."""
        estimated_start_time = None
        for activity in activity_chain:
            estimated_start_time, _ = self._get_timing_from_activity(activity)
            if estimated_start_time:
                return estimated_start_time
        # this can happen only in a chain that is malformed,
        # conaining only Home and S- activities.
        raise sagaexceptions.TripGenerationActivityError(
            f"Missing Primary (P-) activity in the chain {activity_chain}"
        )

    def generate_person_stages(self, from_area, to_area, activity_chain, mode):
        """Returns the trip for the given activity chain."""

        # this estimated start time is going to be used as a tentative departure time
        # for the intermodal routes, it's higly unreliable, but necessary to generate
        # routes with public transports.
        estimated_start_time = self._get_estimated_activity_time_from_chain(
            activity_chain
        )

        # Define a generic Home and Primary activity location.
        from_edge, to_edge = self._stages_define_main_locations(
            from_area, to_area, mode, estimated_start_time
        )

        ## Generate preliminary stages for a person
        person_stages = dict()
        for pos, activity in enumerate(activity_chain):
            if activity not in self._conf["activities"]:
                raise sagaexceptions.TripGenerationActivityError(
                    "Activity {} is not define in the config file.".format(activity)
                )
            _start, _duration = self._get_timing_from_activity(activity)
            if pos == 0:
                if activity != "Home":
                    raise sagaexceptions.TripGenerationActivityError(
                        "Every activity chain MUST start with 'Home', '{}' given.".format(
                            activity
                        )
                    )
                ## Beginning
                person_stages[pos] = self.Activity(
                    activity=activity,
                    fromEdge=from_edge,
                    start=_start,
                    duration=_duration,
                )
            elif "P-" in activity:
                ## This is a primary activity
                person_stages[pos] = self.Activity(
                    activity=activity, toEdge=to_edge, start=_start, duration=_duration
                )
            elif "S-" in activity:
                ## This is a secondary activity
                person_stages[pos] = self.Activity(
                    activity=activity, start=_start, duration=_duration
                )
            elif activity == "Home":
                ## End of the activity chain.
                person_stages[pos] = self.Activity(
                    activity=activity,
                    toEdge=from_edge,
                    start=_start,
                    duration=_duration,
                )

        if len(person_stages) <= 2:
            raise sagaexceptions.TripGenerationActivityError(
                "Invalid activity chain. (Minimal: H -> P-? -> H)", activity_chain
            )

        ## Define secondary activity location
        person_stages = self._stages_define_secondary_locations(
            person_stages, from_edge, to_edge
        )

        ## Remove the initial 'Home' stage and update the from of the second stage.
        person_stages[1] = person_stages[1]._replace(fromEdge=person_stages[0].fromEdge)
        if person_stages[0].start:
            person_stages[1] = person_stages[1]._replace(start=person_stages[0].stage)
        del person_stages[0]

        ## Fixing the 'from' field with a forward chain
        pos = 2
        while pos in person_stages:
            person_stages[pos] = person_stages[pos]._replace(
                fromEdge=person_stages[pos - 1].toEdge
            )
            pos += 1

        ## Compute the real starting time for the activity chain based on ETT and durations
        start = self._stages_compute_start_time(person_stages, mode)
        person_stages[1] = person_stages[1]._replace(start=start)

        ## Define the position of each location in the activity chain.
        person_stages = self._stages_define_locations_position(person_stages)

        ## Final location consistency test
        last_edge = person_stages[1].toEdge
        pos = 2
        while pos in person_stages:
            if person_stages[pos].fromEdge != last_edge:
                raise sagaexceptions.TripGenerationActivityError(
                    "Inconsistency in the locations for the chain of activities.",
                    person_stages,
                )
            last_edge = person_stages[pos].toEdge
            pos += 1

        ## Set the final activity to True
        pos = 1
        while pos in person_stages:
            person_stages[pos] = person_stages[pos]._replace(final=False)
            pos += 1
        person_stages[pos - 1] = person_stages[pos - 1]._replace(final=True)

        return person_stages

    def _stages_compute_start_time(self, person_stages, mode):
        """Compute the real starting time for the activity chain."""

        ## Mode split:
        _mode, _ptype, _vtype = sumoutils.get_intermodal_mode_parameters(
            mode, self._conf["intermodalOptions"]["vehicleAllowedParking"]
        )

        # Find the first 'start' defined.
        pos = 1
        while pos in person_stages:
            if person_stages[pos].start:
                break
            pos += 1

        start = person_stages[pos].start
        while pos in person_stages:
            ett, route = None, None
            try:
                route = self._sumo.simulation.findIntermodalRoute(
                    person_stages[pos].fromEdge,
                    person_stages[pos].toEdge,
                    depart=start,
                    modes=_mode,
                    pType=_ptype,
                    vType=_vtype,
                )
                ett = sumoutils.ett_from_route(route)
            except TraCIException:
                raise sagaexceptions.TripGenerationRouteError(
                    "No solution foud for stage {} and modes {}.".format(
                        pformat(person_stages[pos]), mode
                    )
                )
            if pos - 1 in person_stages:
                if person_stages[pos - 1].duration:
                    ett += person_stages[pos - 1].duration
            start -= ett
            pos -= 1
        return start

    def _get_timing_from_activity(self, activity):
        """Compute start and duration from the activity defined in the config file."""
        start = None
        if self._conf["activities"][activity]["start"]:
            start = self._random_generator.normal(
                loc=self._conf["activities"][activity]["start"]["m"],
                scale=self._conf["activities"][activity]["start"]["s"],
            )
            if start < 0:
                return self._get_timing_from_activity(activity)
        duration = None
        if self._conf["activities"][activity]["duration"]:
            duration = self._random_generator.normal(
                loc=self._conf["activities"][activity]["duration"]["m"],
                scale=self._conf["activities"][activity]["duration"]["s"],
            )
            if duration <= 0:
                return self._get_timing_from_activity(activity)
        return start, duration

    def _get_cached_dist(self, orig, dest):
        cached = self._cache.get((orig, dest))
        if cached is None:
            cached = self._sumo.simulation.findRoute(orig, dest).length
            self._cache[(orig, dest)] = cached
        return cached
