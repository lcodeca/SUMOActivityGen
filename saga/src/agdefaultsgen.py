#!/usr/bin/env python3

""" Generate the default values for the SUMOActivityGen.

    Author: Lara CODECA

    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

import json
import xml.etree.ElementTree


class ActivitygenDefaultGenerator:
    """Generate the default values for SUMOActivityGen."""

    def __init__(self, options):

        self._options = options
        self._config_struct = None
        self._amitran_struct = None
        self._load_configurations()
        self._set_taxi_fleet()
        self._load_odmatrix()
        self._generate_taz()
        self._generate_slices()

    def _load_configurations(self):
        """Load JSON configuration file in a dict."""
        with open(self._options.conf_file, "r") as conf_file:  # pylint: disable=W1514
            self._config_struct = json.loads(conf_file.read())

    def _set_taxi_fleet(self):
        """Setup the taxi fleet."""
        self._config_struct["intermodalOptions"][
            "taxiFleetSize"
        ] = self._options.taxi_fleet

    def _load_odmatrix(self):
        """Load the Amitran XML configuration file."""
        self._amitran_struct = self._parse_xml_file(self._options.amitran_file)

    def _generate_slices(self):
        """Generate population and slices from Amitran definition."""
        population = 0.0
        for odpair in self._amitran_struct:
            population += float(odpair["amount"])

        for odpair in self._amitran_struct:
            perc = round(float(odpair["amount"]) / population, 4)
            if perc <= 0:
                continue
            slice_name = f"{odpair['origin']}_{odpair['destination']}"
            self._config_struct["slices"][slice_name] = {
                "perc": perc,
                "loc_origin": odpair["origin"],
                "loc_primary": odpair["destination"],
                "activityChains": self._config_struct["slices"]["default"][
                    "activityChains"
                ],
            }

        self._config_struct["slices"].pop("default", None)
        self._config_struct["population"]["entities"] = self._options.population

    def _generate_taz(self):
        """Generate TAZ from Amitran definition."""
        for odpair in self._amitran_struct:
            self._config_struct["taz"][odpair["origin"]] = [odpair["origin"]]
            self._config_struct["taz"][odpair["destination"]] = [odpair["destination"]]

    @staticmethod
    def _parse_xml_file(xml_file):
        """Extract all odPair info from an Amitran XML file."""
        xml_tree = xml.etree.ElementTree.parse(xml_file).getroot()
        list_xml = []
        for child in xml_tree.iter("odPair"):
            parsed = {}
            for key, value in child.attrib.items():
                parsed[key] = value
            list_xml.append(parsed)
        return list_xml

    def save_configuration_file(self, filename):
        """Save the configuration file."""
        print(f"Creation of {filename}")
        with open(filename, "w") as outfile:  # pylint: disable=W1514
            outfile.write(json.dumps(self._config_struct, indent=4))
        print(f"{filename} created.")
