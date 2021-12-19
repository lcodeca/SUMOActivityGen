#!/usr/bin/env python3

""" Generate the default Amitran OD-matrix from TAZ weights.

    Author: Lara CODECA

    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

import argparse
import sys

from saga.src.amitranfromtaz import AmitranFromTAZWeightsGenerator


def get_options(cmd_args=None):
    """Argument Parser"""
    parser = argparse.ArgumentParser(
        prog="generateAmitranFromTAZWeights.py",
        usage="%(prog)s [options]",
        description="Generate the default Amitran OD-matrix from TAZ weights.",
    )
    parser.add_argument(
        "--taz-weights",
        type=str,
        dest="taz_file",
        required=True,
        help="Weighted TAZ file (CSV).",
    )
    parser.add_argument(
        "--out",
        type=str,
        dest="output",
        required=True,
        help="OD matrix in Amitran format.",
    )
    parser.add_argument(
        "--density",
        type=float,
        dest="density",
        default=3000.0,
        help="Average population density in square kilometers.",
    )
    return parser.parse_args(cmd_args)


def main(cmd_args):
    """Generate the default Amitran OD-matrix from TAZ weights."""
    options = get_options(cmd_args)
    odmatrix = AmitranFromTAZWeightsGenerator(options)
    odmatrix.save_odmatrix_to_file(options.output)
    print("Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
