#!/usr/bin/env python3

""" SUMO Activity-Based Mobility Generator - Misc SUMO Utils

    Author: Lara CODECA

    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

import os
import sys

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
    import traci.constants as tc
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")


def cost_from_route(route):
    """Compute the route cost."""
    cost = 0.0
    for stage in route:
        cost += stage.cost
    return cost


def ett_from_route(route):
    """Compute the route etimated travel time."""
    ett = 0.0
    for stage in route:
        ett += stage.travelTime
    return ett


def get_intermodal_mode_parameters(mode, parking_requirements):
    """Return the correst TraCI parameters for the requested mode.
    Parameters: _mode, _ptype, _vtype
    """
    if mode == "public":
        return "public", "", ""
    if mode == "bicycle":
        return "bicycle", "", "bicycle"
    if mode == "walk":
        return "", "pedestrian", ""
    if mode == "on-demand":
        return "taxi", "", "on-demand"  # The on-demand parameter needs to be confirmed.
    if mode in parking_requirements:
        return (
            "",
            "",
            mode,
        )  # Required to avoid the exchange point outside the parkingStop
    return "car", "", mode  # Enables the walk from the exchange points to destination


def is_valid_route(mode, route, parking_requirements):
    """Handle findRoute and findIntermodalRoute results."""
    if route is None:
        # traci failed
        return False
    _mode, _ptype, _vtype = get_intermodal_mode_parameters(mode, parking_requirements)
    if not isinstance(route, (list, tuple)):
        # list in until SUMO 1.4.0 included, tuple onward
        # only for findRoute
        if len(route.edges) >= 2:
            return True
    elif _mode == "public":
        for stage in route:
            if stage.line:
                return True
    elif _mode in ("car", "bicycle") or _vtype in parking_requirements:
        for stage in route:
            if stage.type == tc.STAGE_DRIVING and len(stage.edges) >= 2:
                return True
    else:
        for stage in route:
            if len(stage.edges) >= 2:
                return True
    return False
