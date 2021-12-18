#!/usr/bin/env python3

""" SUMO Activity-Based Mobility Generator

    Author: Lara CODECA

    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

import argparse
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

from .src.mobilitygen import MobilityGenerator

if "SUMO_HOME" not in os.environ:
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
