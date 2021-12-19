#!/usr/bin/env python3

""" Extract Taxi Stands from OSM.

    Author: Lara CODECA

    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

import argparse
import os
import sys

from saga.src.taxistands import TaxiStandsFromOSMGenerator

if "SUMO_HOME" not in os.environ:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def get_options(cmd_args=None):
    """Argument Parser"""
    parser = argparse.ArgumentParser(
        prog="generateParkingAreasFromOSM.py",
        usage="%(prog)s [options]",
        description="Extract Taxi Stands from OSM.",
    )
    parser.add_argument(
        "--osm", type=str, dest="osm_file", required=True, help="OSM file."
    )
    parser.add_argument(
        "--net", type=str, dest="net_file", required=True, help="SUMO network file."
    )
    parser.add_argument(
        "--out",
        type=str,
        dest="output",
        required=True,
        help="SUMO taxi stands (parkingArea) file.",
    )
    parser.add_argument(
        "--default-capacity",
        type=int,
        dest="default_capacity",
        default=10,
        help="Default taxi stands capacity if the OSM tag is missing.",
    )
    parser.add_argument(
        "--stand-len",
        type=float,
        dest="stand_len",
        default=10.0,
        help="Parking areas length.",
    )
    parser.add_argument(
        "--stand-angle",
        type=float,
        dest="stand_angle",
        default=45.0,
        help="Parking areas angle.",
    )
    parser.add_argument(
        "--distance-from-intersection",
        type=float,
        dest="intersection_buffer",
        default=10.0,
        help="Buffer area used to avoid having the taxi stand entrance too close "
        "to an intersection.",
    )
    return parser.parse_args(cmd_args)


def main(cmd_args):
    """Extract Parking Areas from OSM."""
    options = get_options(cmd_args)
    stands = TaxiStandsFromOSMGenerator(options)
    stands.stands_generation()
    stands.save_stands_to_file(options.output)
    print("Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
