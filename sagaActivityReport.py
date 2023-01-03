#!/usr/bin/env python3

""" From the Tripinfo file, generate the activities report.

    Author: Lara CODECA

    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

import argparse
import collections
import json
import os
from pprint import pformat
import sys

from lxml import etree
import numpy as np


def get_options(cmd_args=None):
    """Argument Parser."""
    parser = argparse.ArgumentParser(
        prog="{}".format(sys.argv[0]),
        usage="%(prog)s [options]",
        description="SAGA Live Monitoring",
    )
    parser.add_argument(
        "--tripinfo", type=str, required=True, help="SUMO TripInfo file (XML)."
    )
    parser.add_argument("--out", type=str, required=True, help="Output file (CSV).")
    return parser.parse_args(cmd_args)


class SAGAReport(object):
    """SAGA Activities Report"""

    TRIPINFO_SCHEMA = os.path.join(
        os.environ["SUMO_HOME"], "data/xsd/tripinfo_file.xsd"
    )

    def __init__(self, cfg):
        self.tripinfo_file = cfg.tripinfo
        self.output_file = cfg.out

        self.tripinfo = collections.defaultdict(dict)
        self.personinfo = collections.defaultdict(dict)

        self.activity_stats = collections.defaultdict(list)

    def load_tripinfo(self):
        """Load the Tripinfo file"""
        # just in case..
        self.tripinfo = collections.defaultdict(dict)
        self.personinfo = collections.defaultdict(dict)

        tree = None
        try:
            # Faster, but it may fail.
            schema = etree.XMLSchema(file=self.TRIPINFO_SCHEMA)
            parser = etree.XMLParser(schema=schema)
            tree = etree.parse(self.tripinfo_file, parser)
        except etree.XMLSyntaxError as excp:
            print(
                "Unable to use {} schema due to exception {}.".format(
                    self.TRIPINFO_SCHEMA, pformat(excp)
                )
            )
            tree = etree.parse(self.tripinfo_file)

        print("Loading {} tripinfo file.".format(self.tripinfo_file))
        for element in tree.getroot():
            if element.tag == "tripinfo":
                self.tripinfo[element.attrib["id"]] = dict(element.attrib)
            elif element.tag == "personinfo":
                self.personinfo[element.attrib["id"]] = dict(element.attrib)
                stages = []
                for stage in element:
                    stages.append([stage.tag, dict(stage.attrib)])
                self.personinfo[element.attrib["id"]]["stages"] = stages
            else:
                raise Exception("Unrecognized element in the tripinfo file.")
        # print('TRIPINFO: \n{}'.format(self.tripinfo))
        # print('PERSONINFO: \n{}'.format(self.personinfo))

    def process_tripinfo(self):
        """Process the Tripinfo file"""
        print("Processing {} tripinfo file.".format(self.tripinfo_file))
        for person, data in self.personinfo.items():
            for tag, stage in data["stages"]:
                # print('[{}] {} \n{}'.format(person, tag, pformat(stage)))
                if tag == "stop":
                    self.activity_stats[stage["actType"]].append(
                        {
                            "arrival": stage["arrival"],
                            "duration": stage["duration"],
                        }
                    )

    def compute_stats(self):
        """Computing Stats from the Tripinfo file"""
        print("Computing statistics..")
        stats = dict()
        for activity, data in self.activity_stats.items():
            duration = list()
            start = list()
            for value in data:
                start.append(float(value["arrival"]) - float(value["duration"]))
                duration.append(float(value["duration"]))
            stats[activity] = {
                "duration": {
                    "min": min(duration),
                    "max": max(duration),
                    "mean": np.mean(duration),
                    "median": np.median(duration),
                    "std": np.std(duration),
                },
                "start": {
                    "min": min(start),
                    "max": max(start),
                    "mean": np.mean(start),
                    "median": np.median(start),
                    "std": np.std(start),
                },
            }
            print("[{}] \n{}".format(activity, pformat(stats[activity])))

        print("Saving statistics to {} file.".format(self.output_file))
        with open(self.output_file, "w") as output:
            json.dump(stats, output)


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
