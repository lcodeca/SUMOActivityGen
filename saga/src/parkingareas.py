#!/usr/bin/env python3

""" Extract Parking Areas from OSM.

    Author: Lara CODECA

    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

import os
import sys
import xml.etree.ElementTree

from tqdm import tqdm

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
    import sumolib
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


class ParkingAreasFromOSMGenerator:
    """Generate the SUMO additional file for parkings based on OSM."""

    def __init__(self, options):
        self._options = options
        self._osm = self._parse_xml_file(options.osm_file)
        self._net = sumolib.net.readNet(options.net_file)
        self._parkings_edges_dict = {}
        self._osm_parkings = {}
        self._sumo_parkings = {}

    def parkings_generation(self):
        """Main finction to generate all the parking areas."""

        print("Filtering OSM for parking lot..")
        self._filter_parkings()

        print("Create parkings for SUMO..")
        self._parkings_to_edges()
        self._parkings_sumo()

    def save_parkings_to_file(self, filename):
        """Save the generated parkings to file."""
        self._save_parkings_to_file(filename)

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
                if attribute.tag in parsed:
                    parsed[attribute.tag].append(attribute.attrib)
                else:
                    parsed[attribute.tag] = [attribute.attrib]

            if child.tag in dict_xml:
                dict_xml[child.tag].append(parsed)
            else:
                dict_xml[child.tag] = [parsed]
        return dict_xml

    def _filter_parkings(self):
        """Retrieve all the parking lots from a OSM structure."""

        for node in tqdm(self._osm["node"]):
            parking = False
            if "tag" not in list(node.keys()):
                continue
            for tag in node["tag"]:
                if self._is_parkings(tag):
                    parking = True
            if parking:
                x_coord, y_coord = self._net.convertLonLat2XY(node["lon"], node["lat"])
                node["x"] = x_coord
                node["y"] = y_coord
                self._osm_parkings[node["id"]] = node

        print(f"Gathered {len(self._osm_parkings)} parking lots.")

    _PARKING_DICT = {
        "amenity": ["parking", "motorcycle_parking", "parking_entrance"],
        "name": ["underground parking"],
        "parking": ["surface", "underground", "multi-storey"],
        "service": ["parking_aisle"],
    }

    def _is_parkings(self, tag):
        """Check if the tag matches to one of the possible parking lots."""
        for key, value in self._PARKING_DICT.items():
            if tag["k"] == key and tag["v"] in value:
                return True
        return False

    def _parkings_to_edges(self):
        """Associate the parking-id to and edge-id in a dictionary."""
        for parking in tqdm(self._osm_parkings.values()):
            self._parkings_edges_dict[parking["id"]] = self._parking_to_edge(parking)

    def _parking_to_edge(self, parking):
        """Given a parking lot, return the closest edge (lane_0) and all the other info
        required by SUMO for the parking areas:
        (edge_info, lane_info, location, parking.coords, parking.capacity)
        """

        edge_info = None
        lane_info = None
        dist_lane = sys.float_info.max
        location = None

        radius = 50.0
        while not edge_info:
            nearest_edges = self._net.getNeighboringEdges(
                parking["x"], parking["y"], r=radius
            )
            for edge, _ in nearest_edges:
                if not (edge.allows("passenger") and edge.allows("pedestrian")):
                    continue
                if self._is_too_short(edge.getLength()):
                    continue

                # select the lane closer to the curb
                selected_lane = None
                for lane in edge.getLanes():
                    if not lane.allows("passenger"):
                        continue
                    selected_lane = lane
                    break

                if selected_lane is not None:

                    pos, dist = selected_lane.getClosestLanePosAndDist(
                        (float(parking["x"]), float(parking["y"]))
                    )
                    if dist < dist_lane:
                        edge_info = edge
                        lane_info = selected_lane
                        dist_lane = dist
                        location = pos
            radius += 50.0

        if dist_lane > 50.0:
            print(
                f"Alert: parking lots {parking['id']} is {dist_lane} meters from lane {lane_info.getID()}."  # pylint: disable=C0301
            )

        return (edge_info, lane_info, location)

    def _is_too_short(self, edge_len):
        """Check if the edge type is appropriate for a parking lot."""
        if edge_len < (
            self._options.parking_len + 2 * self._options.intersection_buffer
        ):
            return True
        return False

    def _get_capacity(self, parking_id):
        """Retrieve parking lot capacity from OSM."""
        for tag in self._osm_parkings[parking_id]["tag"]:
            if tag["k"] == "capacity":
                try:
                    return int(tag["v"])
                except ValueError:
                    print(
                        f"Parking {parking_id} capacity is not an integer [{tag['v']}]."
                    )
                    return self._options.default_capacity
        print(f"Parking {parking_id} has no capacity tag.")
        return self._options.default_capacity

    def _parkings_sumo(self):
        """Compute the parking lots stops location for SUMO."""
        for plid, (_, lane, location) in self._parkings_edges_dict.items():
            new_pl = {
                "id": plid,
                "lane": lane.getID(),
                "start": location - self._options.parking_len / 2,
                "end": location + self._options.parking_len / 2,
                "capacity": self._get_capacity(plid),
                "coords": (
                    float(self._osm_parkings[plid]["x"]),
                    float(self._osm_parkings[plid]["y"]),
                ),
            }

            if new_pl["start"] < self._options.intersection_buffer:
                new_pl["start"] = self._options.intersection_buffer
                new_pl["end"] = new_pl["start"] + self._options.parking_len

            if new_pl["end"] > lane.getLength() - self._options.intersection_buffer:
                new_pl["end"] = lane.getLength() - self._options.intersection_buffer
                new_pl["start"] = new_pl["end"] - self._options.parking_len

            self._sumo_parkings[plid] = new_pl

    # pylint: disable=C0301
    _ADDITIONALS_TPL = """<?xml version="1.0" encoding="UTF-8"?>

<!-- Generated with generateParkingAreasFromOSM.py [https://github.com/lcodeca/SUMOActivityGen] -->

<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd"> {content}
</additional>
    """

    # pylint: disable=C0301
    _PARKINGS_TPL = """
    <parkingArea id="{id}" lane="{lane}" startPos="{start}" endPos="{end}" roadsideCapacity="{capacity}" friendlyPos="true"/>"""

    def _save_parkings_to_file(self, filename):
        """Save the parking lots into a SUMO XML additional file."""
        print(f"Creation of {filename}")
        with open(filename, "w") as outfile:  # pylint: disable=W1514
            list_of_parkings = ""
            for parking in self._sumo_parkings.values():
                list_of_parkings += self._PARKINGS_TPL.format(
                    id=parking["id"],
                    lane=parking["lane"],
                    start=parking["start"],
                    end=parking["end"],
                    capacity=parking["capacity"],
                )
            content = list_of_parkings
            outfile.write(self._ADDITIONALS_TPL.format(content=content))
        print(f"{filename} created.")