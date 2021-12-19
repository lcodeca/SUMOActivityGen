#!/usr/bin/env python3

""" Generate the default values for the SUMOActivityGen.

    Author: Lara CODECA

    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

import argparse
import sys

from saga.src.agdefaultsgen import ActivitygenDefaultGenerator


def get_options(cmd_args=None):
    """Argument Parser"""
    parser = argparse.ArgumentParser(
        prog="generateDefaultsActivityGen.py",
        usage="%(prog)s [options]",
        description="Generate the default values for the SUMOActivityGen.",
    )
    parser.add_argument(
        "--conf",
        type=str,
        dest="conf_file",
        required=True,
        help="Default configuration file.",
    )
    parser.add_argument(
        "--od-amitran",
        type=str,
        dest="amitran_file",
        required=True,
        help="OD matrix in Amitran format.",
    )
    parser.add_argument(
        "--out", type=str, dest="output", required=True, help="Output file."
    )
    parser.add_argument(
        "--population",
        type=int,
        dest="population",
        default=1000,
        help="Population: number of entities to generate.",
    )
    parser.add_argument(
        "--taxi-fleet",
        type=int,
        dest="taxi_fleet",
        default=10,
        help="Size of the taxi fleet.",
    )
    return parser.parse_args(cmd_args)


def main(cmd_args):
    """Generate the default values for SUMOActivityGen."""
    options = get_options(cmd_args)

    defaults = ActivitygenDefaultGenerator(options)
    defaults.save_configuration_file(options.output)

    print("Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
