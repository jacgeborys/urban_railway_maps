#!/usr/bin/env python3
"""
Multi-City Transit Map Generator - Clean & Comprehensive Version
Uses structured OSM configuration with enhanced stop processing
"""

import requests
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from cities import CITIES
from osm_config import build_overpass_query, classify_route, classify_stop, TRANSFER_CONFIG, VISUAL_CONFIG

# Overpass API endpoint
OVERPASS_URL = "http://overpass-api.de/api/interpreter"


def create_transformer(city_config):
    """Create coordinate transformer for the city"""
    utm_epsg = f"EPSG:326{city_config['utm_zone']:02d}"  # UTM WGS84 North
    return Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)


def fetch_osm_data(center_lat, center_lon, radius_km, query_type):
    """Fetch OSM data using bounding box around coordinates"""
    print(f"Fetching {query_type} data around ({center_lat:.4f}, {center_lon:.4f}) within {radius_km}km...")

    query = build_overpass_query(center_lat, center_lon, radius_km, query_type)

    response = requests.post(OVERPASS_URL, data={"data": query})
    if response.status_code != 200:
        raise RuntimeError(f"Overpass error: {response.status_code}")

    return response.json()


def process_routes(data, transformer, route_category):
    """Process route data for a specific category (tram/metro/train)"""

    # Collect all nodes
    all_nodes = {}
    for el in data["elements"]:
        if el["type"] == "node":
            lon, lat = el["lon"], el["lat"]
            x, y = transformer.transform(lon, lat)
            all_nodes[el["id"]] = (x, y)

    # Build way data
    ways = {}
    for el in data["elements"]:
        if el["type"] == "way" and "nodes" in el:
            ways[el["id"]] = el["nodes"]

    # Collect route coordinates
    route_lines = []

    # Process route relations
    for el in data["elements"]:
        if el["type"] == "relation":
            # Classify this route
            classified_type = classify_route(el)

            # Only include if it matches our target category
            if classified_type == route_category:
                # Get all way members
                for member in el.get("members", []):
                    if member["type"] == "way" and member["ref"] in ways:
                        coords = []
                        for node_id in ways[member["ref"]]:
                            if node_id in all_nodes:
                                coords.append(all_nodes[node_id])
                        if len(coords) >= 2:
                            route_lines.append(coords)

    return route_lines


def process_stops(stops_data, transformer):
    """
    Enhanced stop processing - keeps different transit types separate
    """
    # First pass: collect ALL stops with their raw info
    all_stops = {}  # {name: [{'coord': (x,y), 'type': 'tram/metro/train', 'element': el}, ...]}

    for el in stops_data["elements"]:
        if el["type"] == "node":
            x, y = transformer.transform(el["lon"], el["lat"])
            coord = (x, y)
            name = el["tags"].get("name", "Unnamed")

            # Classify the stop
            stop_type = classify_stop(el)

            if stop_type:  # Only keep classified stops
                # Clean name for grouping (same logic as before)
                base_name = name.split(' 0')[0] if ' 0' in name else name
                base_name = base_name.split(' peron')[0] if ' peron' in base_name else base_name

                if base_name not in all_stops:
                    all_stops[base_name] = []

                all_stops[base_name].append({
                    'coord': coord,
                    'type': stop_type,
                    'element': el,
                    'original_name': name
                })

    # Second pass: resolve conflicts and merge
    tram_stops_dict = {}
    metro_stops_dict = {}
    train_stops_dict = {}

    merge_stats = {
        'same_type_merges': 0,
        'cross_type_merges': 0,
        'conflicts_resolved': 0,
        'transfer_points': 0
    }

    for base_name, stop_list in all_stops.items():
        if len(stop_list) == 1:
            # Single stop, no merging needed
            stop = stop_list[0]
            add_to_category(stop, tram_stops_dict, metro_stops_dict, train_stops_dict)
        else:
            # Multiple stops with same name - need to merge/resolve
            resolved_stops = resolve_stop_conflicts(stop_list, base_name, merge_stats)

            # resolved_stops is now a list (may contain multiple stops)
            for stop in resolved_stops:
                add_to_category(stop, tram_stops_dict, metro_stops_dict, train_stops_dict)

    print(f"Stop processing stats:")
    print(f"  - Same-type merges: {merge_stats['same_type_merges']}")
    print(f"  - Transfer points detected: {merge_stats['transfer_points']}")
    print(f"  - Conflicts resolved: {merge_stats['conflicts_resolved']}")

    # Convert to centroid lists
    def dict_to_centroids(stops_dict):
        centroids = []
        for coords in stops_dict.values():
            if coords:
                avg_x = sum(c[0] for c in coords) / len(coords)
                avg_y = sum(c[1] for c in coords) / len(coords)
                centroids.append((avg_x, avg_y))
        return centroids

    return (
        dict_to_centroids(tram_stops_dict),
        dict_to_centroids(metro_stops_dict),
        dict_to_centroids(train_stops_dict)
    )


def resolve_stop_conflicts(stop_list, base_name, stats):
    """
    Resolve conflicts when multiple stops have the same name - keeps different types separate

    Args:
        stop_list: List of stop dictionaries with same name
        base_name: The cleaned stop name
        stats: Statistics dictionary to update

    Returns:
        List of resolved stops (may be multiple if they should stay separate)
    """
    # Get all unique types
    types = list(set(stop['type'] for stop in stop_list))

    if len(types) == 1:
        # All same type - simple merge
        stats['same_type_merges'] += 1
        merged_coord = calculate_centroid([stop['coord'] for stop in stop_list])
        return [{
            'coord': merged_coord,
            'type': types[0],
            'original_name': base_name
        }]
    else:
        # Multiple types - check if they should be separate or merged
        coords = [stop['coord'] for stop in stop_list]
        max_distance = calculate_max_distance(coords)

        # Group stops by type for potential separate handling
        stops_by_type = {}
        for stop in stop_list:
            stop_type = stop['type']
            if stop_type not in stops_by_type:
                stops_by_type[stop_type] = []
            stops_by_type[stop_type].append(stop)

        if max_distance > 50:  # More than 50m apart - keep separate
            stats['conflicts_resolved'] += 1
            print(f"KEEPING SEPARATE: {base_name} - {types} (distance: {max_distance:.0f}m)")

            # Return separate stops for each type
            result = []
            for stop_type, type_stops in stops_by_type.items():
                merged_coord = calculate_centroid([s['coord'] for s in type_stops])
                result.append({
                    'coord': merged_coord,
                    'type': stop_type,
                    'original_name': base_name
                })
            return result

        else:
            # Close together - potential transfer point, keep separate
            stats['transfer_points'] += 1
            print(f"TRANSFER POINT: {base_name} - {types} (distance: {max_distance:.0f}m)")

            # Return separate stops for each type
            result = []
            for stop_type, type_stops in stops_by_type.items():
                merged_coord = calculate_centroid([s['coord'] for s in type_stops])
                result.append({
                    'coord': merged_coord,
                    'type': stop_type,
                    'original_name': base_name
                })
            return result


def add_to_category(stop, tram_dict, metro_dict, train_dict):
    """Add stop to appropriate category dictionary"""
    coord = stop['coord']
    name = stop['original_name']

    if stop['type'] == 'tram':
        if name not in tram_dict:
            tram_dict[name] = []
        tram_dict[name].append(coord)
    elif stop['type'] == 'metro':
        if name not in metro_dict:
            metro_dict[name] = []
        metro_dict[name].append(coord)
    elif stop['type'] == 'train':
        if name not in train_dict:
            train_dict[name] = []
        train_dict[name].append(coord)


def calculate_centroid(coords):
    """Calculate centroid of coordinate list"""
    avg_x = sum(c[0] for c in coords) / len(coords)
    avg_y = sum(c[1] for c in coords) / len(coords)
    return (avg_x, avg_y)


def calculate_max_distance(coords):
    """Calculate maximum distance between any two coordinates"""
    max_dist = 0
    for i, coord1 in enumerate(coords):
        for coord2 in coords[i + 1:]:
            dist = Point(coord1).distance(Point(coord2))
            max_dist = max(max_dist, dist)

    return max_dist


def find_transfers(tram_stops, metro_stops, train_stops):
    """Find transfer points using configuration-based distances"""

    all_stadiums = []

    # Metro-Tram transfers
    max_dist = TRANSFER_CONFIG['metro_tram']['max_distance']
    for metro_coord in metro_stops:
        metro_point = Point(metro_coord)
        for tram_coord in tram_stops:
            tram_point = Point(tram_coord)
            if metro_point.distance(tram_point) <= max_dist:
                all_stadiums.append([metro_coord, tram_coord])

    # Train-Tram transfers
    max_dist = TRANSFER_CONFIG['train_tram']['max_distance']
    for train_coord in train_stops:
        train_point = Point(train_coord)
        for tram_coord in tram_stops:
            tram_point = Point(tram_coord)
            if train_point.distance(tram_point) <= max_dist:
                all_stadiums.append([train_coord, tram_coord])

    # Train-Metro transfers
    max_dist = TRANSFER_CONFIG['train_metro']['max_distance']
    for train_coord in train_stops:
        train_point = Point(train_coord)
        for metro_coord in metro_stops:
            metro_point = Point(metro_coord)
            if train_point.distance(metro_point) <= max_dist:
                all_stadiums.append([train_coord, metro_coord])

    # Same-type close connections
    def add_close_connections(stops, config_key):
        max_dist = TRANSFER_CONFIG[config_key]['max_distance']
        for i, stop1 in enumerate(stops):
            point1 = Point(stop1)
            for stop2 in stops[i + 1:]:
                point2 = Point(stop2)
                if point1.distance(point2) <= max_dist:
                    all_stadiums.append([stop1, stop2])

    add_close_connections(tram_stops, 'tram_tram')
    add_close_connections(metro_stops, 'metro_metro')
    add_close_connections(train_stops, 'train_train')

    # Create stadium polygons
    individual_stadiums = []
    for stops in all_stadiums:
        if len(stops) == 2:
            stop1, stop2 = stops
            center_x = (stop1[0] + stop2[0]) / 2
            center_y = (stop1[1] + stop2[1]) / 2

            distance = Point(stop1).distance(Point(stop2))
            length = distance + 80
            width = 120

            import math
            angle = math.atan2(stop2[1] - stop1[1], stop2[0] - stop1[0])

            individual_stadiums.append({
                'center': (center_x, center_y),
                'length': length,
                'width': width,
                'angle': angle,
                'stops': stops
            })

    # Create stadium polygons and merge overlapping ones
    def create_stadium_polygon(stadium):
        center = stadium['center']
        length = stadium['length']
        width = stadium['width']
        angle = stadium['angle']

        try:
            radius = width / 2
            rect_length = max(0, length - width)

            if rect_length < 1:
                return Point(center).buffer(radius)

            points = []
            n_points = 8

            # Right semicircle
            for i in range(n_points + 1):
                theta = -np.pi / 2 + np.pi * i / n_points
                x = rect_length / 2 + radius * np.cos(theta)
                y = radius * np.sin(theta)
                points.append([x, y])

            # Left semicircle
            for i in range(n_points + 1):
                theta = np.pi / 2 + np.pi * i / n_points
                x = -rect_length / 2 + radius * np.cos(theta)
                y = radius * np.sin(theta)
                points.append([x, y])

            # Rotate and translate
            points = np.array(points)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_points = points @ rotation_matrix.T
            final_points = rotated_points + np.array(center)

            poly = Polygon(final_points)
            if not poly.is_valid:
                poly = poly.buffer(0)

            return poly

        except Exception as e:
            print(f"Stadium polygon creation failed, using circle: {e}")
            return Point(center).buffer(width / 2)

    # Create and merge polygons
    stadium_polygons = []
    for stadium in individual_stadiums:
        try:
            poly = create_stadium_polygon(stadium)
            if poly and poly.is_valid and not poly.is_empty:
                stadium_polygons.append((poly, stadium))
        except Exception as e:
            print(f"Skipping invalid stadium: {e}")
            continue

    # Group overlapping stadiums
    merged_groups = []
    used_indices = set()

    for i, (poly1, stadium1) in enumerate(stadium_polygons):
        if i in used_indices:
            continue

        group_polys = [poly1]
        group_stadiums = [stadium1]
        used_indices.add(i)

        found_overlap = True
        while found_overlap:
            found_overlap = False
            try:
                current_union = unary_union(group_polys)

                for j, (poly2, stadium2) in enumerate(stadium_polygons):
                    if j in used_indices:
                        continue

                    if current_union.intersects(poly2):
                        group_polys.append(poly2)
                        group_stadiums.append(stadium2)
                        used_indices.add(j)
                        found_overlap = True
            except Exception as e:
                print(f"Error in stadium merging: {e}")
                break

        try:
            merged_union = unary_union(group_polys)
            if merged_union and merged_union.is_valid:
                merged_groups.append((merged_union, group_stadiums))
        except Exception as e:
            print(f"Error creating union: {e}")
            continue

    final_stadiums = []
    for merged_poly, stadiums in merged_groups:
        final_stadiums.append({
            'polygon': merged_poly,
            'stadiums': stadiums
        })

    return final_stadiums


def plot_transit_map(tram_lines, metro_lines, train_lines, tram_stops, metro_stops, train_stops, transfers,
                     city_config):
    """Create the transit map visualization using visual configuration"""

    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot lines using visual config
    for line_type, lines, config_key in [
        ('train', train_lines, 'train'),
        ('metro', metro_lines, 'metro'),
        ('tram', tram_lines, 'tram')
    ]:
        config = VISUAL_CONFIG[config_key]
        for line in lines:
            if len(line) >= 2:
                xs, ys = zip(*line)
                ax.plot(xs, ys,
                        color=config['color'],
                        linewidth=config['linewidth'],
                        alpha=config['alpha'],
                        zorder=config['zorder'])

    # Plot stops using visual config
    for stops, config_key in [
        (train_stops, 'train'),
        (metro_stops, 'metro'),
        (tram_stops, 'tram')
    ]:
        if stops:
            config = VISUAL_CONFIG[config_key]
            xs, ys = zip(*stops)
            ax.scatter(xs, ys,
                       c=config['stop_color'],
                       s=config['stop_size'],
                       marker=config['stop_marker'],
                       zorder=config['zorder'] + 2)

    # Plot transfer stations
    transfer_config = VISUAL_CONFIG['transfer']
    for transfer in transfers:
        merged_poly = transfer['polygon']

        from matplotlib.patches import Polygon as MPLPolygon

        if hasattr(merged_poly, 'geoms'):
            for geom in merged_poly.geoms:
                if hasattr(geom, 'exterior'):
                    coords = list(geom.exterior.coords)
                    patch = MPLPolygon(coords,
                                       facecolor=transfer_config['facecolor'],
                                       edgecolor=transfer_config['edgecolor'],
                                       linewidth=transfer_config['linewidth'],
                                       alpha=transfer_config['alpha'],
                                       zorder=transfer_config['zorder'])
                    ax.add_patch(patch)
        else:
            if hasattr(merged_poly, 'exterior'):
                coords = list(merged_poly.exterior.coords)
                patch = MPLPolygon(coords,
                                   facecolor=transfer_config['facecolor'],
                                   edgecolor=transfer_config['edgecolor'],
                                   linewidth=transfer_config['linewidth'],
                                   alpha=transfer_config['alpha'],
                                   zorder=transfer_config['zorder'])
                ax.add_patch(patch)

    # Set up the plot
    ax.set_aspect('equal')
    ax.axis('off')

    # Set city-specific bounds
    city_center = city_config['center']
    transformer = create_transformer(city_config)
    center_x, center_y = transformer.transform(city_center[1], city_center[0])

    bounds_size = city_config['bounds_km'] * 1000 / 2

    ax.set_xlim(center_x - bounds_size, center_x + bounds_size)
    ax.set_ylim(center_y - bounds_size, center_y + bounds_size)

    # Add city title
    ax.text(0.02, 0.98, city_config['name'], transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


def generate_city_map(city_key):
    """Generate transit map for a specific city"""

    if city_key not in CITIES:
        print(f"City '{city_key}' not found. Available cities: {list(CITIES.keys())}")
        return

    city_config = CITIES[city_key]
    city_name = city_config['name']
    center_lat, center_lon = city_config['center']

    # Use a radius that covers the bounds_km area (convert from square to radius)
    radius_km = city_config['bounds_km'] / 2  # Half the square side length

    print(f"\n=== Generating transit map for {city_name} ===")
    print(f"Center: ({center_lat:.4f}, {center_lon:.4f})")
    print(f"Search radius: {radius_km} km")

    transformer = create_transformer(city_config)

    try:
        # Fetch all data using coordinates
        tram_data = fetch_osm_data(center_lat, center_lon, radius_km, 'tram')
        metro_data = fetch_osm_data(center_lat, center_lon, radius_km, 'metro')
        train_data = fetch_osm_data(center_lat, center_lon, radius_km, 'train')
        stops_data = fetch_osm_data(center_lat, center_lon, radius_km, 'stops')

        # Process routes by category
        tram_lines = process_routes(tram_data, transformer, 'tram')
        metro_lines = process_routes(metro_data, transformer, 'metro')
        train_lines = process_routes(train_data, transformer, 'train')

        # Process stops
        tram_stops, metro_stops, train_stops = process_stops(stops_data, transformer)

        print(f"Found {len(tram_lines)} tram route segments")
        print(f"Found {len(metro_lines)} metro route segments")
        print(f"Found {len(train_lines)} train route segments")
        print(f"Found {len(tram_stops)} tram stops")
        print(f"Found {len(metro_stops)} metro stations")
        print(f"Found {len(train_stops)} train stations")

        # Find transfers
        transfers = find_transfers(tram_stops, metro_stops, train_stops)
        print(f"Found {len(transfers)} transfer stations")

        # Create map
        fig = plot_transit_map(tram_lines, metro_lines, train_lines,
                               tram_stops, metro_stops, train_stops,
                               transfers, city_config)

        # Save map
        filename = f'img/{city_key}_transit_map'
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')

        print(f"Map saved as {filename}.png")

    except Exception as e:
        print(f"Error generating map for {city_name}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function"""
    import sys

    if len(sys.argv) > 1:
        city_key = sys.argv[1].lower()
        generate_city_map(city_key)
    else:
        print("Available cities:")
        for key, config in CITIES.items():
            print(f"  {key}: {config['name']}")

        city_key = input("\nEnter city key (or 'all' for all cities): ").lower().strip()

        if city_key == 'all':
            for city_key in CITIES.keys():
                generate_city_map(city_key)
        else:
            generate_city_map(city_key)


if __name__ == "__main__":
    main()