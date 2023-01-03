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
import xml.etree.ElementTree
from tqdm import tqdm

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
    import sumolib
else:
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


class TaxiStandsFromOSMGenerator:
    """
    Generate the SUMO additional file for taxi stands based on parking areas
    extracted from OSM.
    """

    def __init__(self, options):
        self._options = options
        self._osm = self._parse_xml_file(options.osm_file)
        self._net = sumolib.net.readNet(options.net_file)
        self._stands_edges_dict = dict()
        self._osm_stands = dict()
        self._sumo_stands = dict()

    def stands_generation(self):
        """Main finction to generate all the taxi stands."""

        print("Filtering OSM for taxi stands..")
        self._filter_stands()

        print("Create stands for SUMO..")
        self._stands_to_edges()
        self._stands_sumo()

    def save_stands_to_file(self, filename):
        """Save the generated stands to file."""
        self._save_stands_to_file(filename)

    @staticmethod
    def _parse_xml_file(xml_file):
        """Extract all info from an OSM file."""
        xml_tree = xml.etree.ElementTree.parse(xml_file).getroot()
        dict_xml = {}
        for child in xml_tree:
            parsed = {}
            for key, value in child.attrib.items():
                parsed[key] = value

            for attribute in child:
                if attribute.tag in list(parsed.keys()):
                    parsed[attribute.tag].append(attribute.attrib)
                else:
                    parsed[attribute.tag] = [attribute.attrib]

            if child.tag in list(dict_xml.keys()):
                dict_xml[child.tag].append(parsed)
            else:
                dict_xml[child.tag] = [parsed]
        return dict_xml

    def _filter_stands(self):
        """Retrieve all the taxi stands lots from a OSM structure."""

        for node in tqdm(self._osm["node"]):
            stand = False
            if "tag" not in list(node.keys()):
                continue
            for tag in node["tag"]:
                if self._is_stands(tag):
                    stand = True
            if stand:
                x_coord, y_coord = self._net.convertLonLat2XY(node["lon"], node["lat"])
                node["x"] = x_coord
                node["y"] = y_coord
                self._osm_stands[node["id"]] = node

        print("Gathered {} taxi stands.".format(len(list(self._osm_stands.keys()))))

    _TAXI_STANDS_DICT = {
        "amenity": ["taxi"],
    }

    def _is_stands(self, tag):
        """Check if the tag matches to one of the possible taxi stands."""
        for key, value in self._TAXI_STANDS_DICT.items():
            if tag["k"] == key and tag["v"] in value:
                return True
        return False

    def _stands_to_edges(self):
        """Associate the stand-id to and edge-id in a dictionary."""
        for stand in tqdm(self._osm_stands.values()):
            self._stands_edges_dict[stand["id"]] = self._stand_to_edge(stand)

    def _stand_to_edge(self, stand):
        """Given a taxi stand, return the closest edge (lane_0) and all the other info
        required by SUMO for the parking areas:
        (edge_info, lane_info, location, stand.coords, stand.capacity)
        """

        edge_info = None
        lane_info = None
        dist_lane = sys.float_info.max
        location = None

        radius = 50.0
        while not edge_info:
            nearest_edges = self._net.getNeighboringEdges(
                stand["x"], stand["y"], r=radius
            )
            for edge, _ in nearest_edges:
                if not (edge.allows("taxi") and edge.allows("pedestrian")):
                    continue
                if self._is_too_short(edge.getLength()):
                    continue

                # select the lane closer to the curb
                selected_lane = None
                for lane in edge.getLanes():
                    if not lane.allows("taxi"):
                        continue
                    selected_lane = lane
                    break

                if selected_lane is not None:

                    pos, dist = selected_lane.getClosestLanePosAndDist(
                        (float(stand["x"]), float(stand["y"]))
                    )
                    if dist < dist_lane:
                        edge_info = edge
                        lane_info = selected_lane
                        dist_lane = dist
                        location = pos
            radius += 50.0

        if dist_lane > 50.0:
            print(
                "Alert: taxi stand {} is {} meters from lane {}.".format(
                    stand["id"], dist_lane, lane_info.getID()
                )
            )

        return (edge_info, lane_info, location)

    def _is_too_short(self, edge_len):
        """Check if the edge type is appropriate for a taxi stand."""
        if edge_len < (self._options.stand_len + 2 * self._options.intersection_buffer):
            return True
        return False

    def _get_capacity(self, stand_id):
        """Retrieve taxi stand capacity from OSM."""
        for tag in self._osm_stands[stand_id]["tag"]:
            if tag["k"] == "capacity":
                return int(tag["v"])
        print("Taxi stand {} has no capacity tag.".format(stand_id))
        return self._options.default_capacity

    def _stands_sumo(self):
        """Compute the taxi stands location for SUMO."""
        for plid, (_, lane, location) in self._stands_edges_dict.items():
            new_pl = {
                "id": plid,
                "lane": lane.getID(),
                "start": location - self._options.stand_len / 2,
                "end": location + self._options.stand_len / 2,
                "capacity": self._get_capacity(plid),
                "coords": (
                    float(self._osm_stands[plid]["x"]),
                    float(self._osm_stands[plid]["y"]),
                ),
            }

            if new_pl["start"] < self._options.intersection_buffer:
                new_pl["start"] = self._options.intersection_buffer
                new_pl["end"] = new_pl["start"] + self._options.stand_len

            if new_pl["end"] > lane.getLength() - self._options.intersection_buffer:
                new_pl["end"] = lane.getLength() - self._options.intersection_buffer
                new_pl["start"] = new_pl["end"] - self._options.stand_len

            self._sumo_stands[plid] = new_pl

    _ADDITIONALS_TPL = """<?xml version="1.0" encoding="UTF-8"?>

<!-- Generated with generateParkingAreasFromOSM.py [https://github.com/lcodeca/SUMOActivityGen] -->

<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd"> {content}
</additional>
    """

    _PARKINGS_TPL = """
    <parkingArea id="{id}" lane="{lane}" startPos="{start}" endPos="{end}" roadsideCapacity="{capacity}" friendlyPos="true"/>"""  # pylint: disable=C0301

    def _save_stands_to_file(self, filename):
        """Save the taxi stands into a SUMO XML additional file."""
        print("Creation of {}".format(filename))
        with open(filename, "w") as outfile:
            list_of_stands = ""
            for stand in self._sumo_stands.values():
                list_of_stands += self._PARKINGS_TPL.format(
                    id=stand["id"],
                    lane=stand["lane"],
                    start=stand["start"],
                    end=stand["end"],
                    capacity=stand["capacity"],
                )
            content = list_of_stands
            outfile.write(self._ADDITIONALS_TPL.format(content=content))
        print("{} created.".format(filename))


def main(cmd_args):
    """Extract Parking Areas from OSM."""
    options = get_options(cmd_args)
    stands = TaxiStandsFromOSMGenerator(options)
    stands.stands_generation()
    stands.save_stands_to_file(options.output)
    print("Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
