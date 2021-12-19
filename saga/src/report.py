#!/usr/bin/env python3

""" From the Tripinfo file, generate the activities report.

    Author: Lara CODECA

    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

import collections
import json
import os
from pprint import pformat

from lxml import etree
import numpy as np


class SAGAReport:
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
                f"Unable to use {self.TRIPINFO_SCHEMA} schema due to exception {pformat(excp)}."
            )
            tree = etree.parse(self.tripinfo_file)

        print(f"Loading {self.tripinfo_file} tripinfo file.")
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
        print(f"Processing {self.tripinfo_file} tripinfo file.")
        for data in self.personinfo.values():
            for tag, stage in data["stages"]:
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
        stats = {}
        for activity, data in self.activity_stats.items():
            duration = []
            start = []
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
            print(f"[{activity}] \n{pformat(stats[activity])}")

        print(f"Saving statistics to {self.output_file} file.")
        with open(self.output_file, "w") as output:  # pylint: disable=W1514
            json.dump(stats, output)
