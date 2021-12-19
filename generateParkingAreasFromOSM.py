#!/usr/bin/env python3

""" Extract Parking Areas from OSM.

    Author: Lara CODECA

    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

import argparse
import os
import sys

from saga.src.parkingareas import ParkingAreasFromOSMGenerator

if "SUMO_HOME" not in os.environ:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def get_options(cmd_args=None):
    """Argument Parser"""
    parser = argparse.ArgumentParser(
        prog="generateParkingAreasFromOSM.py",
        usage="%(prog)s [options]",
        description="Extract Parking Areas from OSM.",
    )
    parser.add_argument(
        "--osm", type=str, dest="osm_file", required=True, help="OSM file."
    )
    parser.add_argument(
        "--net", type=str, dest="net_file", required=True, help="SUMO network file."
    )
    parser.add_argument(
        "--out", type=str, dest="output", required=True, help="SUMO parking areas file."
    )
    parser.add_argument(
        "--default-capacity",
        type=int,
        dest="default_capacity",
        default=100,
        help="Default parking areas capacity if the OSM tag is missing.",
    )
    parser.add_argument(
        "--parking-len",
        type=float,
        dest="parking_len",
        default=10.0,
        help="Parking areas length.",
    )
    parser.add_argument(
        "--parking-angle",
        type=float,
        dest="parking_angle",
        default=45.0,
        help="Parking areas angle.",
    )
    parser.add_argument(
        "--distance-from-intersection",
        type=float,
        dest="intersection_buffer",
        default=10.0,
        help="Buffer area used to avoid having the parking entrance "
        "too close to an intersection.",
    )
    return parser.parse_args(cmd_args)


def main(cmd_args):
    """Extract Parking Areas from OSM."""
    options = get_options(cmd_args)
    parkings = ParkingAreasFromOSMGenerator(options)
    parkings.parkings_generation()
    parkings.save_parkings_to_file(options.output)
    print("Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
