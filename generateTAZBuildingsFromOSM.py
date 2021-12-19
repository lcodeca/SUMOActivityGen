#!/usr/bin/env python3

""" Generate TAZ and Buildings weight from OSM.

    Author: Lara CODECA

    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

import argparse
import os
import sys

from saga.src.tazfromosm import GenerateTAZandWeightsFromOSM

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
    import sumolib
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def get_options(cmd_args=None):
    """Argument Parser"""
    parser = argparse.ArgumentParser(
        prog="generateTAZBuildingsFromOSM.py",
        usage="%(prog)s [options]",
        description="Generate TAZ and Buildings weight from OSM.",
    )
    parser.add_argument(
        "--osm", type=str, dest="osm_file", required=True, help="OSM file."
    )
    parser.add_argument(
        "--net", type=str, dest="net_file", required=True, help="SUMO network xml file."
    )
    parser.add_argument(
        "--taz-output",
        type=str,
        dest="taz_output",
        required=True,
        help="TAZ output file (XML).",
    )
    parser.add_argument(
        "--weight-output",
        type=str,
        dest="od_output",
        required=True,
        help="TAZ weight output file (CSV).",
    )
    parser.add_argument(
        "--poly-output",
        type=str,
        dest="poly_output",
        required=True,
        help="Prefix for the POLY output files (CSV).",
    )
    parser.add_argument(
        "--single-taz",
        dest="single_taz",
        action="store_true",
        help="Ignore administrative boundaries and generate only one TAZ.",
    )
    parser.add_argument(
        "--admin-level",
        type=int,
        dest="admin_level",
        default=None,
        help="Select only the administrative boundaries with the given level "
        "and generate the associated TAZs.",
    )
    parser.add_argument(
        "--max-entrance-dist",
        type=float,
        dest="max_entrance",
        default=1000.0,
        help="Maximum search radious to find building eetrances in edges that "
        "are in other TAZs. [Default: 1000.0 meters]",
    )
    parser.add_argument(
        "--taz-plot",
        type=str,
        dest="html_filename",
        default="",
        help="Plots the TAZs to an HTML file as OSM overlay. (Requires folium)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        dest="processes",
        default=1,
        help="Number of processes spawned to associate buildings and edges.",
    )
    parser.set_defaults(single_taz=False)
    return parser.parse_args(cmd_args)


def main(cmd_args):
    """Generate TAZ and Buildings weight from OSM."""
    args = get_options(cmd_args)
    taz_generator = GenerateTAZandWeightsFromOSM(args)
    taz_generator.generate_taz()
    taz_generator.save_sumo_taz(args.taz_output)
    taz_generator.save_taz_weigth(args.od_output)
    taz_generator.generate_buildings()
    taz_generator.save_buildings_weigth(args.poly_output)
    if args.html_filename:
        taz_generator.save_taz_to_osm(args.html_filename)
    print("Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
