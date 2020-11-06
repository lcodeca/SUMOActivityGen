#!/usr/bin/env python3

""" SUMO Activity-Based Mobility Generator - Exceptions

    Author: Lara CODECA

    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

import logging
from pprint import pformat
import sys

logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout)],
                    level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

class TripGenerationGenericError(Exception):
    """
    During the trip generation, various erroroneous states can be reached.
    """
    def __init__(self, message=None):
        """ Init the error message. """
        super().__init__()
        self.message = message
        if self.message:
            logging.debug(self.message)

class TripGenerationActivityError(TripGenerationGenericError):
    """
    During the generation from the activity chains, various erroroneous states can be reached.
    """
    def __init__(self, message=None, activity=None):
        """ Init the error message. """
        super().__init__()
        self.message = message
        self.activity = activity
        if self.message is not None:
            logging.debug(self.message)
        if self.activity is not None:
            with open('TripGenerationActivityError.log', 'a') as openfile:
                openfile.write(message + '\n')
                openfile.write(pformat(activity) + '\n')

class TripGenerationRouteError(TripGenerationGenericError):
    """
    During the step by step generation of the trip, it is possible to reach a state in which
        some of the chosen locations are impossible to reach.
    """
    def __init__(self, message=None, route=None):
        """ Init the error message. """
        super().__init__()
        self.message = message
        self.route = route
        if self.message is not None:
            logging.debug(self.message)
        if self.route is not None:
            with open('TripGenerationRouteError.log', 'a') as openfile:
                openfile.write(message + '\n')
                openfile.write(pformat(route) + '\n')

class TripGenerationInconsistencyError(TripGenerationGenericError):
    """
    During the step by step generation of the trip, it is possible to reach a state in which
        some of the chosen modes are impossible to be used in that order.
    """
    def __init__(self, message=None, plan=None):
        """ Init the error message. """
        super().__init__()
        self.message = message
        self.plan = plan
        if self.message is not None:
            logging.debug(self.message)
        if self.plan is not None:
            with open('TripGenerationInconsistencyError.log', 'a') as openfile:
                openfile.write(message + '\n')
                openfile.write(pformat(plan) + '\n')
