"""
Multi-City Transit Map Generator - Clean & Comprehensive Version
Includes visualization for operational, planned, and disused routes.
"""
import itertools
import requests
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from cities import CITIES
from osm_config import build_overpass_query, classify_route, classify_stop, classify_non_operational_way, VISUAL_CONFIG
from geometry_config import TRANSFER_MAX_DISTANCES, STOP_PROCESSING_CONFIG, TRANSFER_VISUAL_CONFIG

OVERPASS_URL = "http://overpass-api.de/api/interpreter"


def create_transformer(city_config):
    """Create coordinate transformer for the city, handling hemispheres."""
    hemisphere_code = '7' if city_config.get('hemisphere') == 'S' else '6'
    utm_epsg = f"EPSG:32{hemisphere_code}{city_config['utm_zone']:02d}"
    print(f"Using projection: {utm_epsg}")
    return Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)


def fetch_osm_data(center_lat, center_lon, radius_km, query_type):
    """Fetch OSM data using stable, combined queries."""
    print(f"Fetching {query_type} data...")
    query = build_overpass_query(center_lat, center_lon, radius_km, query_type)
    response = requests.post(OVERPASS_URL, data={"data": query})
    if response.status_code != 200:
        raise RuntimeError(f"Overpass error: {response.status_code}\n{response.text}")
    return response.json()


def process_routes(data, transformer):
    """Process raw route data and classify it into separate categories."""
    all_nodes = {el["id"]: transformer.transform(el["lon"], el["lat"]) for el in data["elements"] if
                 el["type"] == "node"}
    ways = {el["id"]: el["nodes"] for el in data["elements"] if el["type"] == "way" and "nodes" in el}
    lines_by_type = {'tram': [], 'metro': [], 'train': [], 'light_rail': []}
    for el in data["elements"]:
        if el["type"] == "relation":
            category = classify_route(el)
            if category in lines_by_type:
                for member in el.get("members", []):
                    if member["type"] == "way" and member["ref"] in ways:
                        coords = [all_nodes[node_id] for node_id in ways[member["ref"]] if node_id in all_nodes]
                        if len(coords) >= 2: lines_by_type[category].append(coords)
    return lines_by_type


def process_non_operational_ways(data, transformer):
    """Process and classify ways that are not operational."""
    lines_by_type = {}
    for element in data.get('elements', []):
        if element.get('type') == 'way':
            style_key = classify_non_operational_way(element)
            if style_key:
                if style_key not in lines_by_type: lines_by_type[style_key] = []
                coords = [(node['lon'], node['lat']) for node in element.get('geometry', [])]
                if len(coords) >= 2:
                    transformed_coords = list(zip(*transformer.transform(*zip(*coords))))
                    lines_by_type[style_key].append(transformed_coords)
    return lines_by_type


def process_stops(stops_data, transformer):
    """Classify all stops and group them by type and name, then find their centroids."""
    all_stops = {}
    for el in stops_data.get("elements", []):
        if el["type"] == "node" and "tags" in el:
            stop_type = classify_stop(el)
            if stop_type:
                name = el["tags"].get("name", "Unnamed")
                base_name = name.split(' 0')[0].split(' peron')[0]
                if base_name not in all_stops: all_stops[base_name] = []
                all_stops[base_name].append(
                    {'coord': transformer.transform(el["lon"], el["lat"]), 'type': stop_type, 'original_name': name})
    stop_dicts = {'tram': {}, 'metro': {}, 'train': {}, 'light_rail': {}}
    for base_name, stop_list in all_stops.items():
        if len(stop_list) == 1:
            add_to_category(stop_list[0], stop_dicts)
        else:
            for stop in resolve_stop_conflicts(stop_list, base_name): add_to_category(stop, stop_dicts)
    return tuple(dict_to_centroids(stop_dicts[t]) for t in ['tram', 'metro', 'train', 'light_rail'])


def resolve_stop_conflicts(stop_list, base_name):
    """Resolve conflicts for multiple stops sharing a name."""
    types = list(set(stop['type'] for stop in stop_list))
    if len(types) == 1: return [
        {'coord': calculate_centroid([s['coord'] for s in stop_list]), 'type': types[0], 'original_name': base_name}]
    max_distance = calculate_max_distance([s['coord'] for s in stop_list])
    if max_distance > STOP_PROCESSING_CONFIG['max_conflict_distance']:
        print(f"KEEPING SEPARATE: {base_name} ({types}) - distance: {max_distance:.0f}m")
    else:
        print(f"TRANSFER POINT: {base_name} ({types}) - distance: {max_distance:.0f}m")
    result = []
    for stop_type in types:
        type_stops = [s for s in stop_list if s['type'] == stop_type]
        result.append({'coord': calculate_centroid([s['coord'] for s in type_stops]), 'type': stop_type,
                       'original_name': base_name})
    return result


def add_to_category(stop, stop_dicts):
    """Helper to add a stop to the correct category dictionary."""
    stop_type, name = stop['type'], stop['original_name']
    if name not in stop_dicts[stop_type]: stop_dicts[stop_type][name] = []
    stop_dicts[stop_type][name].append(stop['coord'])


def calculate_centroid(coords):
    """Calculate centroid of a coordinate list."""
    return (sum(c[0] for c in coords) / len(coords), sum(c[1] for c in coords) / len(coords))


def calculate_max_distance(coords):
    """Calculate maximum distance between any two coordinates in a list."""
    if len(coords) < 2: return 0
    return max(Point(c1).distance(Point(c2)) for i, c1 in enumerate(coords) for c2 in coords[i + 1:])


def dict_to_centroids(stops_dict):
    """Convert a dictionary of named stops to a list of their centroids."""
    return [calculate_centroid(coords) for coords in stops_dict.values() if coords]


def find_transfers(stop_lists):
    """Find transfer points and create merged stadium polygons."""
    all_stadiums, type_names = [], ['tram', 'light_rail', 'metro', 'train']
    for (i, type1), (j, type2) in itertools.combinations_with_replacement(enumerate(type_names), 2):
        stops1, stops2, key = stop_lists[i], stop_lists[j], '_'.join(sorted([type1, type2]))
        max_dist = TRANSFER_MAX_DISTANCES.get(key)
        if not max_dist or not stops1 or not stops2: continue
        source_list = itertools.product(stops1, stops2) if i != j else itertools.combinations(stops1, 2)
        for coord1, coord2 in source_list:
            if Point(coord1).distance(Point(coord2)) <= max_dist: all_stadiums.append([coord1, coord2])
    stadium_polygons = [p for p in [create_stadium_polygon(s) for s in all_stadiums] if
                        p and p.is_valid and not p.is_empty]
    if not stadium_polygons: return []
    merged_union = unary_union([p.buffer(0) for p in stadium_polygons])
    return list(merged_union.geoms) if hasattr(merged_union, 'geoms') else [merged_union]


def create_stadium_polygon(stops):
    """Creates a proper stadium-shaped polygon around two stop coordinates."""
    try:
        coord1, coord2 = stops;
        p1, p2 = Point(coord1), Point(coord2);
        distance = p1.distance(p2)
        if distance < 1: return p1.buffer(TRANSFER_VISUAL_CONFIG['stadium_width'] / 2)
        center_x, center_y = (p1.x + p2.x) / 2, (p1.y + p2.y) / 2
        length, width, radius = distance + TRANSFER_VISUAL_CONFIG['stadium_length_padding'], TRANSFER_VISUAL_CONFIG[
            'stadium_width'], TRANSFER_VISUAL_CONFIG['stadium_width'] / 2
        angle = np.arctan2(p2.y - p1.y, p2.x - p1.x);
        rect_half_length = max(0, length - width) / 2
        points = np.array([(-rect_half_length, radius), (rect_half_length, radius), (rect_half_length, -radius),
                           (-rect_half_length, -radius)])
        cos_a, sin_a = np.cos(angle), np.sin(angle);
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        final_points = (points @ rotation_matrix.T) + np.array([center_x, center_y]);
        stadium_rect = Polygon(final_points)
        cap_center1_local, cap_center2_local = np.array([rect_half_length, 0]), np.array([-rect_half_length, 0])
        cap1_center = Point((cap_center1_local @ rotation_matrix.T) + np.array([center_x, center_y]))
        cap2_center = Point((cap_center2_local @ rotation_matrix.T) + np.array([center_x, center_y]))
        stadium_poly = unary_union([stadium_rect, cap1_center.buffer(radius), cap2_center.buffer(radius)])
        return stadium_poly.buffer(0) if not stadium_poly.is_valid else stadium_poly
    except Exception as e:
        print(f"Stadium polygon creation failed: {e}");
        return None


def plot_transit_map(lines_by_type, stops_by_type, non_operational_lines, transfers, city_config):
    """Create the transit map visualization from categorized data."""
    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot non-operational lines first, using their specific styles
    for style_key, lines in non_operational_lines.items():
        if style_key in VISUAL_CONFIG:
            config = VISUAL_CONFIG[style_key]
            for line in lines:
                if len(line) >= 2:
                    xs, ys = zip(*line)
                    ax.plot(xs, ys, **config)

    # Plot operational lines and stops
    for line_type in ['train', 'metro', 'light_rail', 'tram']:
        config = VISUAL_CONFIG[line_type]
        for line in lines_by_type.get(line_type, []):
            xs, ys = zip(*line);
            ax.plot(xs, ys, color=config['color'], linewidth=config['linewidth'], alpha=config['alpha'],
                    zorder=config['zorder'])
        if stops_by_type.get(line_type):
            xs, ys = zip(*stops_by_type[line_type]);
            ax.scatter(xs, ys, c=config['stop_color'], s=config['stop_size'], marker=config['stop_marker'],
                       zorder=config['zorder'] + 5)

    # Plot transfer stadiums on top
    from matplotlib.patches import Polygon as MPLPolygon
    transfer_config = {k: v for k, v in VISUAL_CONFIG['transfer'].items() if k != 'zorder'}
    zorder = VISUAL_CONFIG['transfer']['zorder']
    for poly in transfers:
        if hasattr(poly, 'exterior'):
            ax.add_patch(MPLPolygon(list(poly.exterior.coords), zorder=zorder, **transfer_config))

    ax.set_aspect('equal');
    ax.axis('off')
    center_x, center_y = create_transformer(city_config).transform(city_config['center'][1], city_config['center'][0])
    bounds_size = city_config['bounds_km'] * 1000 / 2
    ax.set_xlim(center_x - bounds_size, center_x + bounds_size);
    ax.set_ylim(center_y - bounds_size, center_y + bounds_size)
    ax.text(0.02, 0.98, city_config['name'], transform=ax.transAxes, fontsize=20, fontweight='bold', va='top',
            ha='left',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    plt.tight_layout();
    return fig


def generate_city_map(city_key):
    """Orchestrates the fetching, processing, and plotting for a city."""
    if city_key not in CITIES: print(f"City '{city_key}' not found."); return
    city_config = CITIES[city_key]
    city_name, center_lat, center_lon = city_config['name'], city_config['center'][0], city_config['center'][1]
    radius_km = city_config['bounds_km'] / 2
    print(f"\n=== Generating transit map for {city_name} ===");
    transformer = create_transformer(city_config)
    try:
        # Fetch all data
        tram_light_rail_data = fetch_osm_data(center_lat, center_lon, radius_km, 'tram_and_light_rail')
        metro_data = fetch_osm_data(center_lat, center_lon, radius_km, 'metro')
        train_data = fetch_osm_data(center_lat, center_lon, radius_km, 'train')
        stops_data = fetch_osm_data(center_lat, center_lon, radius_km, 'stops')
        non_operational_data = fetch_osm_data(center_lat, center_lon, radius_km, 'non_operational')

        # Process all data
        lines_by_type = {'tram': [], 'metro': [], 'train': [], 'light_rail': []}
        for data in [tram_light_rail_data, metro_data, train_data]:
            for key, lines in process_routes(data, transformer).items(): lines_by_type[key].extend(lines)

        non_operational_lines = process_non_operational_ways(non_operational_data, transformer)

        tram_stops, metro_stops, train_stops, light_rail_stops = process_stops(stops_data, transformer)
        stops_by_type = {'tram': tram_stops, 'metro': metro_stops, 'train': train_stops, 'light_rail': light_rail_stops}

        # Print stats
        for t in ['tram', 'light_rail', 'metro', 'train']: print(
            f"Found {len(lines_by_type[t])} operational {t} segments and {len(stops_by_type[t])} stops.")
        print(f"Found {sum(len(v) for v in non_operational_lines.values())} non-operational segments.")

        transfers = find_transfers(list(stops_by_type.values()));
        print(f"Found {len(transfers)} transfer areas.")

        # Plot the map
        fig = plot_transit_map(lines_by_type, stops_by_type, non_operational_lines, transfers, city_config)

        filename = f'img/{city_key}_transit_map.png';
        plt.savefig(filename, dpi=300, bbox_inches='tight');
        print(f"Map saved as {filename}")

    except Exception as e:
        print(f"Error generating map for {city_name}: {e}");
        import traceback;
        traceback.print_exc()


def main():
    """Main entry point to run the map generator."""
    import sys, os
    if not os.path.exists('img'): os.makedirs('img')
    if len(sys.argv) > 1:
        city_key = sys.argv[1].lower()
        if city_key == 'all':
            [generate_city_map(key) for key in CITIES.keys()]
        else:
            generate_city_map(city_key)
    else:
        print("Available cities:");
        [print(f"  {key}: {config['name']}") for key, config in CITIES.items()]
        try:
            city_key = input("\nEnter city key (or 'all'): ").lower().strip()
            if city_key:
                if city_key == 'all':
                    [generate_city_map(key) for key in CITIES.keys()]
                else:
                    generate_city_map(city_key)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")


if __name__ == "__main__":
    main()