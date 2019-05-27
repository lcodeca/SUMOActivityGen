#!/usr/bin/env python3

""" Generate TAZ and Buildings weight from OSM.

    Author: Lara CODECA

    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

import argparse
import csv
import logging
import os
import sys
import xml.etree.ElementTree

from functools import partial
import pyproj
import numpy

import shapely.geometry as geometry
from shapely.ops import transform

from tqdm import tqdm

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    import sumolib
    from sumolib.miscutils import euclidean
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

def logs():
    """ Log init. """
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(handlers=[stdout_handler], level=logging.WARN,
                        format='[%(asctime)s] %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

def get_options(cmd_args=None):
    """ Argument Parser """
    parser = argparse.ArgumentParser(
        prog='generateTAZBuildingsFromOSM.py', usage='%(prog)s [options]',
        description='Generate TAZ and Buildings weight from OSM.')
    parser.add_argument('--osm', type=str, dest='osm_file', required=True,
                        help='OSM file.')
    parser.add_argument('--net', type=str, dest='net_file', required=True,
                        help='SUMO network xml file.')
    parser.add_argument('--taz-output', type=str, dest='taz_output', required=True,
                        help='Prefix for the TAZ output file (XML).')
    parser.add_argument('--od-output', type=str, dest='od_output', required=True,
                        help='Prefix for the OD output file (CSV).')
    parser.add_argument('--poly-output', type=str, dest='poly_output', required=True,
                        help='Prefix for the POLY output files (CSV).')
    return parser.parse_args(cmd_args)

class GenerateTAZandWeightsFromOSM(object):
    """ Generate TAZ and Buildings weight from OSM."""

    _osm = None
    _net = None

    _osm_boundaries = {
        'relation': {},
        'way': {},
        'node': {},
    }

    _osm_buildings = {
        'relation': {},
        'way': {},
        'node': {},
    }

    _taz = dict()

    def __init__(self, osm, net):

        self._osm = osm
        self._net = net

        logging.info('Filtering administrative boudaries from OSM..')
        self._filter_boundaries_from_osm()

        logging.info("Extracting TAZ from OSM-like boundaries.")
        self._build_taz_from_osm()

        logging.info("Computing TAZ areas...")
        self._taz_areas()

    def generate_taz(self):
        """ Generate TAZ by filtering edges,
            additionally computing TAZ weight through nodes and area. """

        logging.info("Filtering edges...")
        self._edges_filter()

        logging.info("Filtering nodes...")
        self._nodes_filter()

    def generate_buildings(self):
        """ Generate the buildings weight with edge and TAZ association."""

        logging.info("Filtering buildings...")
        self._filter_buildings_from_osm()

        logging.info("Sorting buildings in the TAZ...")
        self._sort_buildings()

    def save_sumo_taz(self, filename):
        """ Save TAZ to file. """
        logging.info("Creation of %s", filename)
        self._write_taz_file(filename)

    def save_taz_weigth(self, filename):
        """ Save weigths to file."""
        logging.info("Creation of %s", filename)
        self._write_csv_file(filename)

    def save_buildings_weigth(self, filename):
        """ Save building weights to file."""
        logging.info("Creation of %s", filename)
        self._write_poly_files(filename)

    @staticmethod
    def _is_boundary(tags):
        """ Check tags to find {'k': 'boundary', 'v': 'administrative'} """
        for tag in tags:
            if tag['k'] == 'boundary' and tag['v'] == 'administrative':
                return True
        return False

    def _filter_boundaries_from_osm(self):
        """ Extract boundaries from OSM structure. """

        for relation in tqdm(self._osm['relation']):
            if self._is_boundary(relation['tag']):
                self._osm_boundaries['relation'][relation['id']] = relation
                for member in relation['member']:
                    self._osm_boundaries[member['type']][member['ref']] = {}

        for way in tqdm(self._osm['way']):
            if way['id'] in self._osm_boundaries['way'].keys():
                self._osm_boundaries['way'][way['id']] = way
                for ndid in way['nd']:
                    self._osm_boundaries['node'][ndid['ref']] = {}

        for node in tqdm(self._osm['node']):
            if node['id'] in self._osm_boundaries['node'].keys():
                self._osm_boundaries['node'][node['id']] = node

        logging.info('Found %d administrative boundaries.',
                     len(self._osm_boundaries['relation'].keys()))

    def _build_taz_from_osm(self):
        """ Extract TAZ from OSM boundaries. """

        for id_boundary, boundary in tqdm(self._osm_boundaries['relation'].items()):

            if not boundary:
                logging.critical('Empty boundary %s', id_boundary)
                continue

            list_of_nodes = []
            if 'member' in boundary:
                for member in boundary['member']:
                    if member['type'] == 'way':
                        if 'nd' in self._osm_boundaries['way'][member['ref']]:
                            for node in self._osm_boundaries['way'][member['ref']]['nd']:
                                coord = self._osm_boundaries['node'][node['ref']]
                                list_of_nodes.append((float(coord['lon']), float(coord['lat'])))

            name = None
            ref = None
            for tag in boundary['tag']:
                if tag['k'] == 'name':
                    name = tag['v']
                elif tag['k'] == 'ref':
                    ref = tag['v']
            if not name:
                name = id_boundary
            if not ref:
                ref = id_boundary
            self._taz[id_boundary] = {
                'name': name,
                'ref': ref,
                'convex_hull': geometry.MultiPoint(list_of_nodes).convex_hull,
                'raw_points': geometry.MultiPoint(list_of_nodes),
                'edges': set(),
                'nodes': set(),
                'buildings': set(),
                'buildings_cumul_area': 0,
            }

        logging.info('Generaated %d TAZ.', len(self._taz.keys()))

    def _taz_areas(self):
        """ Compute the area in "shape" for each TAZ """

        for id_taz in tqdm(self._taz.keys()):
            x_coords, y_coords = self._taz[id_taz]['convex_hull'].exterior.coords.xy
            length = len(x_coords)
            poly = []
            for pos in range(length):
                x_coord, y_coord = self._net.convertLonLat2XY(x_coords[pos], y_coords[pos])
                poly.append((x_coord, y_coord))
            self._taz[id_taz]['area'] = geometry.Polygon(poly).area

    def _edges_filter(self):
        """ Sort edges to the right TAZ """
        for edge in tqdm(self._net.getEdges()):
            for coord in edge.getShape():
                lon, lat = self._net.convertXY2LonLat(coord[0], coord[1])
                for id_taz in list(self._taz.keys()):
                    if self._taz[id_taz]['convex_hull'].contains(geometry.Point(lon, lat)):
                        self._taz[id_taz]['edges'].add(edge.getID())

    def _nodes_filter(self):
        """ Sort nodes to the right TAZ """
        for node in tqdm(self._osm['node']):
            for id_taz in list(self._taz.keys()):
                if self._taz[id_taz]['convex_hull'].contains(
                        geometry.Point(float(node['lon']), float(node['lat']))):
                    self._taz[id_taz]['nodes'].add(node['id'])

    @staticmethod
    def _is_building(way):
        """ Return if a way is a building """
        if 'tag' not in way:
            return False
        for tag in way['tag']:
            if tag['k'] == 'building':
                return True
        return False

    def _filter_buildings_from_osm(self):
        """ Extract buildings from OSM structure. """

        for way in tqdm(self._osm['way']):
            if not self._is_building(way):
                continue
            self._osm_buildings['way'][way['id']] = way
            for ndid in way['nd']:
                self._osm_buildings['node'][ndid['ref']] = {}

        for node in tqdm(self._osm['node']):
            if node['id'] in self._osm_buildings['node'].keys():
                self._osm_buildings['node'][node['id']] = node

        logging.info('Found %d buildings.',
                     len(self._osm_buildings['way'].keys()))

    def _get_centroid(self, way):
        """ Return lat lon of the centroid. """
        for tag in way['tag']:
            if tag['k'] == 'centroid':
                splitted = tag['v'].split(',')
                return splitted[0], splitted[1]

        ## the centroid has not yet been computed.
        points = []
        for node in way['nd']:
            points.append([float(self._osm_buildings['node'][node['ref']]['lat']),
                           float(self._osm_buildings['node'][node['ref']]['lon'])])
        centroid = numpy.mean(numpy.array(points), axis=0)

        self._osm_buildings['way'][way['id']]['tag'].append({
            'k':  'centroid',
            'v':  '{}, {}'.format(centroid[0], centroid[1])
            })
        return centroid[0], centroid[1]

    def _get_approx_area(self, way):
        """ Return approximated area of the building. """
        for tag in way['tag']:
            if tag['k'] == 'approx_area':
                return float(tag['v'])

        ## the approx_area has not yet been computed.
        points = []
        for node in way['nd']:
            points.append([float(self._osm_buildings['node'][node['ref']]['lat']),
                           float(self._osm_buildings['node'][node['ref']]['lon'])])

        approx = geometry.MultiPoint(points).convex_hull
        # http://openstreetmapdata.com/info/projections
        proj = partial(pyproj.transform, pyproj.Proj(init='epsg:4326'),
                       pyproj.Proj(init='epsg:3857'))

        converted_approximation = transform(proj, approx)

        self._osm_buildings['way'][way['id']]['tag'].append({
            'k':  'approx_area',
            'v':  converted_approximation.area
            })
        return converted_approximation.area

    def _building_to_edge(self, coords, id_taz):
        """ Given the coords of a building, return te closest edge """

        centroid = coords = (float(coords[0]), float(coords[1]))

        pedestrian_edge_info = None
        pedestrian_dist_edge = sys.float_info.max # distance.euclidean(a,b)

        generic_edge_info = None
        generic_dist_edge = sys.float_info.max # distance.euclidean(a,b)

        for id_edge in self._taz[id_taz]['edges']:
            edge = self._net.getEdge(id_edge)
            if edge.allows('rail'):
                continue
            _, _, dist = edge.getClosestLanePosDist(centroid)
            if edge.allows('passenger') and dist < generic_dist_edge:
                generic_edge_info = edge
                generic_dist_edge = dist
            if edge.allows('pedestrian') and dist < pedestrian_dist_edge:
                pedestrian_edge_info = edge
                pedestrian_dist_edge = dist

        if generic_edge_info and generic_dist_edge > 500.0:
            logging.info("A building entrance [passenger] is %d meters away.",
                         generic_dist_edge)
        if pedestrian_edge_info and pedestrian_dist_edge > 500.0:
            logging.info("A building entrance [pedestrian] is %d meters away.",
                         pedestrian_dist_edge)

        return generic_edge_info, pedestrian_edge_info

    def _sort_buildings(self):
        """ Sort buildings to the right TAZ based on centroid. """

        for _, way in tqdm(self._osm_buildings['way'].items()):
            ## compute the centroid
            lat, lon = self._get_centroid(way)

            ## compute the approximated area
            area = int(self._get_approx_area(way))

            for id_taz in list(self._taz.keys()):
                if self._taz[id_taz]['convex_hull'].contains(
                        geometry.Point(float(lon), float(lat))):
                    generic_edge, pedestrian_edge = self._building_to_edge(
                        self._net.convertLonLat2XY(lon, lat), id_taz)
                    if generic_edge or pedestrian_edge:
                        gen_id = None
                        ped_id = None
                        if generic_edge:
                            gen_id = generic_edge.getID()
                        if pedestrian_edge:
                            ped_id = pedestrian_edge.getID()
                        self._taz[id_taz]['buildings'].add((way['id'], area, gen_id, ped_id))
                        self._taz[id_taz]['buildings_cumul_area'] += area

    _TAZS = """
<tazs> {list_of_tazs}
</tazs>
"""

    _TAZ = """
    <!-- id="{taz_id}" name="{taz_name}" -->
    <taz id="{taz_id}" edges="{list_of_edges}"/>"""

    def _write_taz_file(self, filename):
        """ Write the SUMO file. """
        with open(filename, 'w') as outfile:
            string_of_tazs = ''
            for value in self._taz.values():
                string_of_edges = ''
                for edge in value['edges']:
                    string_of_edges += str(edge) + ' '
                string_of_edges = string_of_edges.strip()
                string_of_tazs += self._TAZ.format(
                    taz_id=value['ref'], taz_name=value['name'], #.encode('utf-8'),
                    list_of_edges=string_of_edges)
            outfile.write(self._TAZS.format(list_of_tazs=string_of_tazs))

    def _write_poly_files(self, prefix):
        """ Write the CSV file. """
        for value in self._taz.values():
            filename = '{}.{}.csv'.format(prefix, value['ref'])
            with open(filename, 'w') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow(['TAZ', 'Poly', 'Area', 'Weight', 'GenEdge', 'PedEdge'])
                for poly, area, g_edge, p_edge in value['buildings']:
                    csvwriter.writerow([value['ref'], poly, area,
                                        area/value['buildings_cumul_area'], g_edge, p_edge])

    def _write_csv_file(self, filename):
        """ Write the CSV file. """
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(['TAZ', 'Name', '#Nodes', 'Area'])
            for value in self._taz.values():
                csvwriter.writerow([value['ref'], value['name'], len(value['nodes']),
                                    value['area']])

def _parse_xml_file(xml_file):
    """ Extract all info from an OSM file. """
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

def main(cmd_args):
    """ Generate TAZ and Buildings weight from OSM. """

    args = get_options(cmd_args)

    osm = _parse_xml_file(args.osm_file)
    net = sumolib.net.readNet(args.net_file)

    taz_generator = GenerateTAZandWeightsFromOSM(osm, net)
    taz_generator.generate_taz()
    taz_generator.save_sumo_taz(args.taz_output)
    taz_generator.save_taz_weigth(args.od_output)
    taz_generator.generate_buildings()
    taz_generator.save_buildings_weigth(args.poly_output)

    logging.info("Done.")

if __name__ == "__main__":
    logs()
    main(sys.argv[1:])
