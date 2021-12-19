#!/usr/bin/env python3

""" From the Tripinfo file, generate the activities report.

    Author: Lara CODECA

    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

import argparse
import sys

from saga.src.report import SAGAReport


def get_options(cmd_args=None):
    """Argument Parser."""
    parser = argparse.ArgumentParser(
        prog=f"{sys.argv[0]}",
        usage="%(prog)s [options]",
        description="SAGA Live Monitoring",
    )
    parser.add_argument(
        "--tripinfo", type=str, required=True, help="SUMO TripInfo file (XML)."
    )
    parser.add_argument("--out", type=str, required=True, help="Output file (CSV).")
    return parser.parse_args(cmd_args)


def main(cmd_args):
    """SAGA Activities Report"""

    args = get_options(cmd_args)
    print(args)

    report = SAGAReport(args)
    report.load_tripinfo()
    report.process_tripinfo()
    report.compute_stats()
    print("Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
