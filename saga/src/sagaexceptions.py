#!/usr/bin/env python3

""" SUMO Activity-Based Mobility Generator - Exceptions

    Author: Lara CODECA

    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

import logging
import json
from pprint import pformat
import sys

LOGGER = logging.getLogger("ActivityGen")


class TripGenerationGenericError(Exception):
    """
    During the trip generation, various erroroneous states can be reached.
    """

    def __init__(self, message=None):
        """Init the error message."""
        super().__init__()
        self.message = message
        if self.message:
            LOGGER.debug(self.message)


class TripGenerationActivityError(TripGenerationGenericError):
    """
    During the generation from the activity chains, various erroroneous states can be reached.
    """

    def __init__(self, message, activity=None):
        """Init the error message."""
        super().__init__()
        self.message = message
        self.activity = activity
        LOGGER.debug(self.message)
        if self.activity is not None:
            with open("TripGenerationActivityError.log", "a") as openfile:
                # openfile.write(message + '\n')
                json.dump(
                    {
                        "msg": message,
                        "activity": activity,
                    },
                    openfile,
                    indent=2,
                )
                # openfile.write(pformat(activity) + '\n')


class TripGenerationRouteError(TripGenerationGenericError):
    """
    During the step by step generation of the trip, it is possible to reach a state in which
        some of the chosen locations are impossible to reach.
    """

    def __init__(self, message, route=None):
        """Init the error message."""
        super().__init__()
        self.message = message
        self.route = route
        LOGGER.debug(self.message)
        if self.route is not None:
            with open("TripGenerationRouteError.log", "a") as openfile:
                # openfile.write(message + '\n')
                json.dump(
                    {
                        "msg": message,
                        "route": route,
                    },
                    openfile,
                    indent=2,
                )
                # openfile.write(pformat(route) + '\n')


class TripGenerationInconsistencyError(TripGenerationGenericError):
    """
    During the step by step generation of the trip, it is possible to reach a state in which
        some of the chosen modes are impossible to be used in that order.
    """

    def __init__(self, message, plan=None):
        """Init the error message."""
        super().__init__()
        self.message = message
        self.plan = plan
        LOGGER.debug(self.message)
        if self.plan is not None:
            with open("TripGenerationInconsistencyError.log", "a") as openfile:
                # openfile.write(message + '\n')
                json.dump(
                    {
                        "msg": message,
                        "plan": plan,
                    },
                    openfile,
                    indent=2,
                )
                # openfile.write(pformat(plan) + '\n')
