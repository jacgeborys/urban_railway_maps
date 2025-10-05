"""
Multi-City Transit Map Generator - Clean & Comprehensive Version
Includes visualization for operational, planned, and disused routes.
"""
import itertools
import requests
import matplotlib.pyplot as plt
import numpy as np
import time
from pyproj import Transformer
from shapely.geometry import Point, Polygon, LineString, MultiLineString
from shapely.ops import unary_union
from cities import CITIES
from osm_config import build_overpass_query, classify_route, classify_stop, classify_non_operational_way, VISUAL_CONFIG
from geometry_config import TRANSFER_MAX_DISTANCES, STOP_PROCESSING_CONFIG, TRANSFER_VISUAL_CONFIG

# Register Inter font
from matplotlib import font_manager
font_regular_path = r"D:\QGIS\functional_map\script\font\Inter_28pt-Regular.ttf"
font_manager.fontManager.addfont(font_regular_path)

OVERPASS_URL = "http://overpass-api.de/api/interpreter"

# Configuration for Overpass API retries
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 5


def create_transformer(city_config):
    """Create coordinate transformer for the city, handling hemispheres."""
    hemisphere_code = '7' if city_config.get('hemisphere') == 'S' else '6'
    utm_epsg = f"EPSG:32{hemisphere_code}{city_config['utm_zone']:02d}"
    print(f"Using projection: {utm_epsg}")
    return Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)


def fetch_osm_data(center_lat, center_lon, radius_km, query_type):
    """
    Fetch OSM data using stable, combined queries with comprehensive retry logic.
    Handles rate limiting (429), server timeouts (504), and other errors gracefully.
    """
    query = build_overpass_query(center_lat, center_lon, radius_km, query_type)

    retries = 0
    while retries < MAX_RETRIES:
        try:
            # Add a small delay before each request to be respectful to the API
            if retries > 0:
                base_delay = INITIAL_RETRY_DELAY * (2 ** (retries - 1))
                actual_delay = base_delay + (retries * 2)  # Extra progressive delay
                print(f"    Waiting {actual_delay} seconds before retry {retries + 1}/{MAX_RETRIES} for {query_type}...")
                time.sleep(actual_delay)
            elif query_type != 'tram_and_light_rail':  # Don't delay the first query type
                time.sleep(2)  # Small delay between different query types

            response = requests.post(OVERPASS_URL, data={"data": query}, timeout=60)

            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as json_error:
                    print(f"    JSON parsing error for {query_type}: {json_error}")
                    retries += 1
                    continue

            elif response.status_code == 429:
                # Rate limiting - be more conservative with delays
                delay = max(INITIAL_RETRY_DELAY * (2 ** retries), 30)  # Minimum 30 seconds for rate limits
                print(f"    Overpass API rate limit (429) for {query_type}. Waiting {delay} seconds (Retry {retries + 1}/{MAX_RETRIES})...")
                time.sleep(delay)
                retries += 1

            elif response.status_code == 504:
                # Gateway timeout - server is overloaded, be patient
                delay = max(INITIAL_RETRY_DELAY * (2 ** retries), 20)  # Minimum 20 seconds for timeouts
                print(f"    Overpass API server timeout (504) for {query_type}. Server is busy, waiting {delay} seconds (Retry {retries + 1}/{MAX_RETRIES})...")
                time.sleep(delay)
                retries += 1

            elif response.status_code in [500, 502, 503]:
                # Server errors - wait and retry
                delay = INITIAL_RETRY_DELAY * (2 ** retries)
                print(f"    Overpass API server error ({response.status_code}) for {query_type}. Retrying in {delay} seconds (Retry {retries + 1}/{MAX_RETRIES})...")
                time.sleep(delay)
                retries += 1

            else:
                # Other HTTP errors - don't retry these
                error_text = response.text[:500] if len(response.text) > 500 else response.text
                raise RuntimeError(f"Overpass API error {response.status_code} for {query_type}:\n{error_text}")

        except requests.exceptions.Timeout:
            delay = INITIAL_RETRY_DELAY * (2 ** retries)
            print(f"    Request timeout for {query_type}. Retrying in {delay} seconds (Retry {retries + 1}/{MAX_RETRIES})...")
            time.sleep(delay)
            retries += 1

        except requests.exceptions.ConnectionError as e:
            delay = INITIAL_RETRY_DELAY * (2 ** retries)
            print(f"    Connection error for {query_type}: {e}. Retrying in {delay} seconds (Retry {retries + 1}/{MAX_RETRIES})...")
            time.sleep(delay)
            retries += 1

        except requests.exceptions.RequestException as e:
            delay = INITIAL_RETRY_DELAY * (2 ** retries)
            print(f"    Network error for {query_type}: {e}. Retrying in {delay} seconds (Retry {retries + 1}/{MAX_RETRIES})...")
            time.sleep(delay)
            retries += 1

    # If we've exhausted all retries
    raise RuntimeError(f"Failed to fetch {query_type} data after {MAX_RETRIES} retries. "
                      f"The Overpass API server may be overloaded. Try again later or consider "
                      f"reducing the search radius for this city.")

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
    """Process and classify ways that are not operational (e.g., construction, proposed)."""
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


def process_stops(stops_data, transformer, operational_lines_buffer_geom): # operational_lines_buffer_geom is no longer used here for filtering
    """
    Classify all stops, group them by type and name, find their centroids.
    Proximity filtering is now handled in generate_city_map at a later stage.
    """
    all_stops = {}
    for el in stops_data.get("elements", []):
        if el["type"] == "node" and "tags" in el:
            stop_type = classify_stop(el)
            if stop_type:
                coord = transformer.transform(el["lon"], el["lat"])
                # REMOVED: Proximity filter from here. Stops are collected regardless of line proximity.
                # if operational_lines_buffer_geom is None or Point(coord).intersects(operational_lines_buffer_geom):
                name = el["tags"].get("name", "Unnamed")
                base_name = name.split(' 0')[0].split(' peron')[0]
                if base_name not in all_stops: all_stops[base_name] = []
                all_stops[base_name].append(
                    {'coord': coord, 'type': stop_type, 'original_name': name})
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
        pass
    else:
        pass
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
    if not coords: return None
    return (sum(c[0] for c in coords) / len(coords), sum(c[1] for c in coords) / len(coords))


def calculate_max_distance(coords):
    """Calculate maximum distance between any two coordinates in a list."""
    if len(coords) < 2: return 0
    return max(Point(c1).distance(Point(c2)) for i, c1 in enumerate(coords) for c2 in coords[i + 1:])


def dict_to_centroids(stops_dict):
    """Convert a dictionary of named stops to a list of their centroids."""
    return [centroid for coords in stops_dict.values() if coords for centroid in [calculate_centroid(coords)] if centroid is not None]


def find_transfers(stop_lists):
    """
    Find transfer points and create merged stadium polygons for *actual transfers only*.
    This function explicitly does NOT generate polygons for single, isolated stops.
    """
    all_stadiums, type_names = [], ['tram', 'light_rail', 'metro', 'train']
    for (i, type1), (j, type2) in itertools.combinations_with_replacement(enumerate(type_names), 2):
        stops1, stops2, key = stop_lists[i], stop_lists[j], '_'.join(sorted([type1, type2]))
        max_dist = TRANSFER_MAX_DISTANCES.get(key)
        if not max_dist: continue

        # For same-type, only consider if there are *multiple* stops to potentially merge
        if i == j and len(stops1) < 2: continue

        if i != j: # Different types: always look for transfers
            source_list = itertools.product(stops1, stops2)
        else: # Same type: only look for transfers if max_dist allows merging distinct points
            source_list = itertools.combinations(stops1, 2)

        for coord1, coord2 in source_list:
            if Point(coord1).distance(Point(coord2)) <= max_dist:
                all_stadiums.append([coord1, coord2])

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
        if distance < 1:
            return p1.buffer(TRANSFER_VISUAL_CONFIG['stadium_width'] / 2)
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
        print(f"Stadium polygon creation failed: {stops} - {e}");
        return None


def plot_transit_map(lines_by_type, stops_by_type, non_operational_lines, transfer_polygons, city_config):
    """Create the transit map visualization from categorized data."""
    from matplotlib import font_manager

    # Create font properties using the Inter font files
    font_regular_path = r"D:\QGIS\functional_map\script\font\Inter_28pt-Regular.ttf"
    font_bold_path = r"D:\QGIS\functional_map\script\font\Inter_28pt-Bold.ttf"

    legend_font = font_manager.FontProperties(fname=font_regular_path, size=12)
    title_font = font_manager.FontProperties(fname=font_bold_path, size=21)

    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot non-operational lines first, using their specific styles
    for style_key, lines in non_operational_lines.items():
        if style_key in VISUAL_CONFIG:
            config = VISUAL_CONFIG[style_key]
            for line in lines:
                if len(line) >= 2:
                    xs, ys = zip(*line)
                    ax.plot(xs, ys, **config)

    # Create a merged geometry of all transfer polygons for efficient checking
    merged_transfer_area = unary_union(transfer_polygons).buffer(
        1) if transfer_polygons else None  # Small buffer for robust intersection check

    # Plot operational lines and individual non-transfer stops
    for line_type in ['train', 'metro', 'light_rail', 'tram']:
        config = VISUAL_CONFIG[line_type]
        for line in lines_by_type.get(line_type, []):
            xs, ys = zip(*line);
            ax.plot(xs, ys, color=config['color'], linewidth=config['linewidth'], alpha=config['alpha'],
                    zorder=config['zorder'])

        # Plot individual stops (circles) only if they are not part of a larger transfer polygon
        # 'stops_by_type' passed here already contains proximity-filtered stops
        if stops_by_type.get(line_type):
            uncovered_stops = []
            for stop_coord in stops_by_type[line_type]:
                stop_point = Point(stop_coord)
                if merged_transfer_area is None or not merged_transfer_area.intersects(stop_point):
                    uncovered_stops.append(stop_coord)

            if uncovered_stops:
                xs, ys = zip(*uncovered_stops);
                ax.scatter(xs, ys, c=config['stop_color'],
                           s=config['stop_size'],
                           marker=config['stop_marker'],
                           alpha=config['stop_alpha'],
                           zorder=config['stop_zorder'],
                           edgecolor='none')

    # Plot transfer polygons on top
    from matplotlib.patches import Polygon as MPLPolygon
    transfer_polygon_style = {
        'facecolor': VISUAL_CONFIG['transfer']['facecolor'],
        'edgecolor': VISUAL_CONFIG['transfer']['edgecolor'],
        'linewidth': VISUAL_CONFIG['transfer']['linewidth'],
        'alpha': VISUAL_CONFIG['transfer']['alpha'],
    }
    zorder = VISUAL_CONFIG['transfer']['zorder']

    for poly in transfer_polygons:
        if hasattr(poly, 'exterior'):
            ax.add_patch(MPLPolygon(list(poly.exterior.coords), zorder=zorder, **transfer_polygon_style))
        elif hasattr(poly, 'geoms'):  # Handle MultiPolygon if any
            for p in poly.geoms:
                if hasattr(p, 'exterior'):
                    ax.add_patch(MPLPolygon(list(p.exterior.coords), zorder=zorder, **transfer_polygon_style))

    ax.set_aspect('equal');
    ax.axis('off')
    map_center_utm = create_transformer(city_config).transform(city_config['center'][1], city_config['center'][0])
    center_x, center_y = map_center_utm[0], map_center_utm[1]
    bounds_size = city_config['bounds_km'] * 1000 / 2
    ax.set_xlim(center_x - bounds_size, center_x + bounds_size);
    ax.set_ylim(center_y - bounds_size, center_y + bounds_size)

    # City title
    ax.text(0.02, 0.98, city_config['name'], transform=ax.transAxes, fontproperties=title_font, fontweight='bold',
            va='top',
            ha='left', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # Copyright attribution at bottom
    copyright_font = font_manager.FontProperties(fname=font_regular_path, size=10)
    ax.text(0.5, 0.01, '© Jacek Gęborys, OpenStreetMap contributors',
            transform=ax.transAxes, fontproperties=copyright_font,
            va='bottom', ha='center', color='#666666', alpha=0.8)

    # Scale bar (2km) in bottom right corner
    scale_length_m = 2000  # 2km in meters
    # Position in axis coordinates
    scale_x_end = 0.98
    scale_y = 0.02

    # Convert to data coordinates for the scale bar
    # Get current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Calculate scale bar position in data coordinates
    data_x_end = xlim[0] + (xlim[1] - xlim[0]) * scale_x_end
    data_x_start = data_x_end - scale_length_m
    data_y = ylim[0] + (ylim[1] - ylim[0]) * scale_y

    # Draw scale bar
    ax.plot([data_x_start, data_x_end], [data_y, data_y],
            color='#333333', linewidth=2, solid_capstyle='butt', zorder=100)

    # Add tick marks at ends
    tick_height = scale_length_m * 0.03
    ax.plot([data_x_start, data_x_start], [data_y - tick_height, data_y + tick_height],
            color='#333333', linewidth=2, zorder=100)
    ax.plot([data_x_end, data_x_end], [data_y - tick_height, data_y + tick_height],
            color='#333333', linewidth=2, zorder=100)

    # Add label
    scale_font = font_manager.FontProperties(fname=font_regular_path, size=9)
    ax.text((data_x_start + data_x_end) / 2, data_y + tick_height * 2, '2 km',
            fontproperties=scale_font, ha='center', va='bottom', color='#333333', zorder=100)

    # Legend in bottom left corner
    legend_elements = []
    legend_labels = []

    # Only add legend items for transit types that actually exist in this city
    legend_order = ['train', 'metro', 'light_rail', 'tram']
    for transit_type in legend_order:
        if lines_by_type.get(transit_type) or stops_by_type.get(transit_type):
            config = VISUAL_CONFIG[transit_type]
            legend_elements.append(plt.Line2D([0], [0], color=config['color'],
                                              linewidth=config['linewidth'] * 1.5))  # Slightly thicker for visibility
            legend_labels.append(transit_type.replace('_', ' ').title())

    if legend_elements:
        legend = ax.legend(legend_elements, legend_labels,
                           loc='lower left',
                           bbox_to_anchor=(0.02, 0.02),
                           frameon=True,
                           fancybox=True,
                           shadow=True,
                           framealpha=0.9,
                           facecolor='white',
                           prop=legend_font)

        # Set legend frame properties
        legend.get_frame().set_linewidth(0.5)
        legend.get_frame().set_edgecolor('gray')

    plt.tight_layout();
    return fig


def generate_city_map(city_key, generate_clean=True):
    """
    Orchestrates the fetching, processing, and plotting for a city.

    Args:
        city_key: The city identifier from CITIES dict
        generate_clean: If True, also generates clean version (without labels) for collages
    """
    if city_key not in CITIES:
        print(f"City '{city_key}' not found.")
        return

    city_config = CITIES[city_key]
    city_name, center_lat, center_lon = city_config['name'], city_config['center'][0], city_config['center'][1]
    radius_km = city_config['bounds_km'] / 2
    print(f"\n=== Generating transit map for {city_name} ===")

    transformer = create_transformer(city_config)

    try:
        start_time = time.time()

        # --- Fetching Data ---
        fetch_start = time.time()
        tram_light_rail_data = fetch_osm_data(center_lat, center_lon, radius_km, 'tram_and_light_rail')
        metro_data = fetch_osm_data(center_lat, center_lon, radius_km, 'metro')
        train_data = fetch_osm_data(center_lat, center_lon, radius_km, 'train')
        stops_data = fetch_osm_data(center_lat, center_lon, radius_km, 'stops')
        non_operational_data = fetch_osm_data(center_lat, center_lon, radius_km, 'non_operational')
        print(f"  Data Fetching complete in {time.time() - fetch_start:.2f} seconds.")

        # --- Processing Operational Lines ---
        process_lines_start = time.time()
        lines_by_type = {'tram': [], 'metro': [], 'train': [], 'light_rail': []}
        for data_source in [tram_light_rail_data, metro_data, train_data]:
            for key, lines in process_routes(data_source, transformer).items():
                lines_by_type[key].extend(lines)
        print(f"  Operational Lines Processing complete in {time.time() - process_lines_start:.2f} seconds.")

        # --- Creating Operational Lines Buffer for Stop Filtering ---
        buffer_start = time.time()
        all_operational_line_geoms = []
        line_simplification_tolerance = STOP_PROCESSING_CONFIG['line_simplification_tolerance']

        for line_type in lines_by_type:
            for segment_coords in lines_by_type[line_type]:
                if len(segment_coords) >= 2:
                    line_geom = LineString(segment_coords)
                    if line_simplification_tolerance > 0:
                        line_geom = line_geom.simplify(line_simplification_tolerance, preserve_topology=True)
                    all_operational_line_geoms.append(line_geom)

        operational_lines_buffer_geom = None
        if all_operational_line_geoms:
            merged_lines = unary_union(all_operational_line_geoms)
            operational_lines_buffer_geom = merged_lines.buffer(STOP_PROCESSING_CONFIG['stop_proximity_buffer'])
        print(f"  Operational Lines Buffer creation complete in {time.time() - buffer_start:.2f} seconds.")

        # --- Processing Non-Operational Lines ---
        non_op_start = time.time()
        non_operational_lines = process_non_operational_ways(non_operational_data, transformer)
        print(f"  Non-Operational Lines Processing complete in {time.time() - non_op_start:.2f} seconds.")

        # --- Processing Stops (raw, without proximity filter yet) ---
        process_stops_start = time.time()
        tram_stops_raw, metro_stops_raw, train_stops_raw, light_rail_stops_raw = process_stops(stops_data, transformer,
                                                                                               operational_lines_buffer_geom)
        stops_by_type_raw = {'tram': tram_stops_raw, 'metro': metro_stops_raw, 'train': train_stops_raw,
                             'light_rail': light_rail_stops_raw}
        print(f"  Raw Stops Processing complete in {time.time() - process_stops_start:.2f} seconds.")

        # --- Filtering Stops at the 'final plotting level' (as requested) ---
        filter_stops_start = time.time()
        filtered_stops_by_type = {'tram': [], 'metro': [], 'train': [], 'light_rail': []}
        for line_type in stops_by_type_raw:
            if operational_lines_buffer_geom:
                for stop_coord in stops_by_type_raw[line_type]:
                    if Point(stop_coord).intersects(operational_lines_buffer_geom):
                        filtered_stops_by_type[line_type].append(stop_coord)
            else:
                filtered_stops_by_type[line_type].extend(stops_by_type_raw[line_type])

        stops_by_type_for_plotting = filtered_stops_by_type
        print(f"  Stops Filtering complete in {time.time() - filter_stops_start:.2f} seconds.")

        # Print stats (using filtered counts)
        for t in ['tram', 'light_rail', 'metro', 'train']:
            print(
                f"  Found {len(lines_by_type[t])} operational {t} segments and {len(stops_by_type_for_plotting[t])} stops (after proximity filter).")
        print(f"  Found {sum(len(v) for v in non_operational_lines.values())} non-operational segments.")

        # --- Finding Transfer Polygons (only actual merges from *filtered* stops) ---
        find_polygons_start = time.time()
        transfer_polygons = find_transfers(list(stops_by_type_for_plotting.values()))
        print(f"  Transfer Polygons generation complete in {time.time() - find_polygons_start:.2f} seconds.")
        print(f"  Found {len(transfer_polygons)} transfer areas.")

        # --- Generate CLEAN map first (no labels) ---
        plot_start = time.time()
        fig = plot_transit_map_clean(lines_by_type, stops_by_type_for_plotting, non_operational_lines,
                                     transfer_polygons, city_config)

        # Save clean version if requested
        if generate_clean:
            filename_clean = f'img/clean/{city_key}_transit_map_clean.png'
            plt.savefig(filename_clean, dpi=300, bbox_inches='tight', pad_inches=0)
            print(f"  Clean map saved as {filename_clean}")

        # --- Add decorative elements on top for labeled version ---
        add_map_labels(fig, city_config, lines_by_type, stops_by_type_for_plotting)

        # Save labeled version
        filename = f'img/{city_key}_transit_map.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Labeled map saved as {filename} in {time.time() - plot_start:.2f} seconds.")


        # Close figure
        plt.close(fig)

        print(f"Total time for {city_name}: {time.time() - start_time:.2f} seconds.")

    except Exception as e:
        print(f"Error generating map for {city_name}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Ensure any remaining figures are closed
        plt.close('all')


def add_map_labels(fig, city_config, lines_by_type, stops_by_type):
    """Add title, legend, copyright, and scale bar to an existing clean map figure."""
    from matplotlib import font_manager

    # Load fonts
    font_regular_path = r"D:\QGIS\functional_map\script\font\Inter_28pt-Regular.ttf"
    font_bold_path = r"D:\QGIS\functional_map\script\font\Inter_28pt-Bold.ttf"

    legend_font = font_manager.FontProperties(fname=font_regular_path, size=12)
    title_font = font_manager.FontProperties(fname=font_bold_path, size=21)
    copyright_font = font_manager.FontProperties(fname=font_regular_path, size=10)
    scale_font = font_manager.FontProperties(fname=font_regular_path, size=9)

    ax = fig.axes[0]  # Get the axis from the existing figure

    # City title
    ax.text(0.02, 0.98, city_config['name'], transform=ax.transAxes, fontproperties=title_font,
            va='top', ha='left', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # Copyright attribution at bottom
    ax.text(0.5, 0.01, '© Jacek Gęborys, OpenStreetMap contributors',
            transform=ax.transAxes, fontproperties=copyright_font,
            va='bottom', ha='center', color='#666666', alpha=0.8)

    # Scale bar (2km) in bottom right corner
    scale_length_m = 2000  # 2km in meters
    scale_x_end = 0.98
    scale_y = 0.02

    # Get current axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Calculate scale bar position in data coordinates
    data_x_end = xlim[0] + (xlim[1] - xlim[0]) * scale_x_end
    data_x_start = data_x_end - scale_length_m
    data_y = ylim[0] + (ylim[1] - ylim[0]) * scale_y

    # Draw scale bar
    ax.plot([data_x_start, data_x_end], [data_y, data_y],
            color='#333333', linewidth=2, solid_capstyle='butt', zorder=100)

    # Add tick marks at ends
    tick_height = scale_length_m * 0.03
    ax.plot([data_x_start, data_x_start], [data_y - tick_height, data_y + tick_height],
            color='#333333', linewidth=2, zorder=100)
    ax.plot([data_x_end, data_x_end], [data_y - tick_height, data_y + tick_height],
            color='#333333', linewidth=2, zorder=100)

    # Add scale label
    ax.text((data_x_start + data_x_end) / 2, data_y + tick_height * 2, '2 km',
            fontproperties=scale_font, ha='center', va='bottom', color='#333333', zorder=100)

    # Legend in bottom left corner
    legend_elements = []
    legend_labels = []

    # Only add legend items for transit types that actually exist in this city
    legend_order = ['train', 'metro', 'light_rail', 'tram']
    for transit_type in legend_order:
        if lines_by_type.get(transit_type) or stops_by_type.get(transit_type):
            config = VISUAL_CONFIG[transit_type]
            legend_elements.append(plt.Line2D([0], [0], color=config['color'],
                                              linewidth=config['linewidth'] * 1.5))
            legend_labels.append(transit_type.replace('_', ' ').title())

    if legend_elements:
        legend = ax.legend(legend_elements, legend_labels,
                           loc='lower left',
                           bbox_to_anchor=(0.02, 0.02),
                           frameon=True,
                           fancybox=True,
                           shadow=True,
                           framealpha=0.9,
                           facecolor='white',
                           prop=legend_font)

        # Set legend frame properties
        legend.get_frame().set_linewidth(0.5)
        legend.get_frame().set_edgecolor('gray')

    plt.tight_layout()

def plot_transit_map_clean(lines_by_type, stops_by_type, non_operational_lines, transfer_polygons, city_config):
    """Create a clean transit map visualization without titles, legends, or labels - for collages."""
    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot non-operational lines first, using their specific styles
    for style_key, lines in non_operational_lines.items():
        if style_key in VISUAL_CONFIG:
            config = VISUAL_CONFIG[style_key]
            for line in lines:
                if len(line) >= 2:
                    xs, ys = zip(*line)
                    ax.plot(xs, ys, **config)

    # Create a merged geometry of all transfer polygons for efficient checking
    merged_transfer_area = unary_union(transfer_polygons).buffer(1) if transfer_polygons else None

    # Plot operational lines and individual non-transfer stops
    for line_type in ['train', 'metro', 'light_rail', 'tram']:
        config = VISUAL_CONFIG[line_type]
        for line in lines_by_type.get(line_type, []):
            xs, ys = zip(*line)
            ax.plot(xs, ys, color=config['color'], linewidth=config['linewidth'], alpha=config['alpha'],
                    zorder=config['zorder'])

        # Plot individual stops (circles) only if they are not part of a larger transfer polygon
        if stops_by_type.get(line_type):
            uncovered_stops = []
            for stop_coord in stops_by_type[line_type]:
                stop_point = Point(stop_coord)
                if merged_transfer_area is None or not merged_transfer_area.intersects(stop_point):
                    uncovered_stops.append(stop_coord)

            if uncovered_stops:
                xs, ys = zip(*uncovered_stops)
                ax.scatter(xs, ys, c=config['stop_color'],
                           s=config['stop_size'],
                           marker=config['stop_marker'],
                           alpha=config['stop_alpha'],
                           zorder=config['stop_zorder'],
                           edgecolor='none')

    # Plot transfer polygons on top
    from matplotlib.patches import Polygon as MPLPolygon
    transfer_polygon_style = {
        'facecolor': VISUAL_CONFIG['transfer']['facecolor'],
        'edgecolor': VISUAL_CONFIG['transfer']['edgecolor'],
        'linewidth': VISUAL_CONFIG['transfer']['linewidth'],
        'alpha': VISUAL_CONFIG['transfer']['alpha'],
    }
    zorder = VISUAL_CONFIG['transfer']['zorder']

    for poly in transfer_polygons:
        if hasattr(poly, 'exterior'):
            ax.add_patch(MPLPolygon(list(poly.exterior.coords), zorder=zorder, **transfer_polygon_style))
        elif hasattr(poly, 'geoms'):
            for p in poly.geoms:
                if hasattr(p, 'exterior'):
                    ax.add_patch(MPLPolygon(list(p.exterior.coords), zorder=zorder, **transfer_polygon_style))

    ax.set_aspect('equal')
    ax.axis('off')

    # Set map bounds
    map_center_utm = create_transformer(city_config).transform(city_config['center'][1], city_config['center'][0])
    center_x, center_y = map_center_utm[0], map_center_utm[1]
    bounds_size = city_config['bounds_km'] * 1000 / 2
    ax.set_xlim(center_x - bounds_size, center_x + bounds_size)
    ax.set_ylim(center_y - bounds_size, center_y + bounds_size)

    # NO title, NO legend, NO copyright, NO scale bar

    plt.tight_layout(pad=0)  # Remove padding
    return fig

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