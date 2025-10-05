"""
OSM Configuration for Transit Map Generator - Final Robust Version
"""
import math
from route_exceptions import get_route_exception

def create_bbox_from_center(center_lat, center_lon, radius_km):
    """Create a bounding box around center coordinates, robust for all hemispheres."""
    if radius_km < 0: radius_km = 0
    lat_offset = radius_km / 111.0
    lon_offset = radius_km / (111.0 * math.cos(math.radians(center_lat)))
    south, north = center_lat - lat_offset, center_lat + lat_offset
    west, east = center_lon - lon_offset, center_lon + lon_offset
    return min(south, north), min(west, east), max(south, north), max(west, east)

def build_overpass_query(center_lat, center_lon, radius_km, query_type):
    """Builds Overpass queries using a global [bbox:...] setting."""
    south, west, north, east = create_bbox_from_center(center_lat, center_lon, radius_km)
    bbox_str = f"{south:.6f},{west:.6f},{north:.6f},{east:.6f}"
    base_query_settings = f"[out:json][timeout:180][bbox:{bbox_str}];"

    if query_type == 'tram_and_light_rail':
        return f"""{base_query_settings}
        (relation["type"="route"]["route"="tram"]["state"!="proposed"];
         relation["type"="route"]["route"="light_rail"]["state"!="proposed"][!"network:metro"];);
        out body; >; out skel qt;"""
    elif query_type == 'metro':
        return f"""{base_query_settings}
        (relation["type"="route"]["route"~"^(subway|metro)$"];);
        out body; >; out skel qt;"""
    elif query_type == 'train':
        return f"""{base_query_settings}
        (relation["type"="route"]["route"="train"]["service"!~"^(international|long_distance)$"];
         relation["type"="route"]["route"="monorail"];
         relation["type"="route"]["route"="light_rail"]["network:metro"];);
        out body; >; out skel qt;"""
    elif query_type == 'stops':
        return f"""{base_query_settings}
        (node["railway"~"^(tram_stop|station|stop|halt)$"];
         node["public_transport"~"^(station|stop_position)$"];
         node["train"="yes"]; node["tram"="yes"]; node["subway"="yes"]; node["light_rail"="yes"];);
        out body;"""
    elif query_type == 'non_operational':
        return f"""{base_query_settings}
        way["railway"~"^(construction|proposed)$"];
        out geom;"""
    else:
        raise ValueError(f"Unknown query type: {query_type}")


def classify_route(route_element):
    """
    Classifies an operational route element into a transit category.
    This now checks for exceptions first, then the 'state' tag to avoid
    misclassifying planned routes.
    """
    tags = route_element.get('tags', {})
    relation_id = route_element.get('id')

    # Check for exceptions first
    exception = get_route_exception(relation_id)
    if exception is not False:
        return exception  # Returns None (exclude) or a specific type (reclassify)

    # Check explicit state tags
    if tags.get('state') in ['proposed', 'construction', 'disused', 'abandoned']:
        return None

    # Classify operational routes
    route = tags.get('route', '')
    if route == 'tram':
        return 'tram'
    if route == 'light_rail':
        return 'train' if tags.get('network:metro') else 'light_rail'
    if route in ['subway', 'metro']:
        return 'metro'
    if route in ['train', 'monorail']:
        return 'train'

    return None


def classify_stop(stop_element):
    """Classifies a stop element into a transit category."""
    tags = stop_element.get('tags', {})

    # Exclude entrances
    if tags.get('railway') == 'subway_entrance':
        return None
    if tags.get('railway') == 'train_station_entrance':
        return None

    # Prioritize explicit mode tags over railway infrastructure tags
    if tags.get('subway') == 'yes' or tags.get('station') == 'subway':
        return 'metro'

    # Check light_rail=yes BEFORE checking railway=tram_stop
    if tags.get('light_rail') == 'yes':
        return 'light_rail'

    # Only classify as tram if explicitly tram=yes (not just railway=tram_stop)
    if tags.get('tram') == 'yes':
        return 'tram'

    # railway=tram_stop without explicit mode tags -> tram
    if tags.get('railway') == 'tram_stop' and tags.get('light_rail') != 'yes':
        return 'tram'

    if tags.get('train') == 'yes':
        return 'train'
    if tags.get('railway') in ['station', 'stop', 'halt']:
        return 'train'

    return None

def classify_non_operational_way(way_element):
    """
    Classifies a non-operational way (construction, proposed)
    and returns a combined key for styling, e.g., 'proposed_tram'.
    """
    tags = way_element.get('tags', {})
    status = tags.get('railway')

    if status not in ['construction', 'proposed']:
        return None

    # Check multiple possible tag patterns for the railway type
    railway_type = (tags.get(f'{status}:railway') or  # construction:railway=subway
                   tags.get(status) or                  # construction=subway
                   tags.get('railway'))                 # railway=construction (not useful)

    if railway_type in ['subway', 'metro'] or tags.get('network', '').startswith('metro'):
        return f'{status}_metro'
    if railway_type == 'tram':
        return f'{status}_tram'
    if railway_type == 'light_rail':
        return f'{status}_light_rail'
    if railway_type == 'train':
        return f'{status}_train'
    return None

VISUAL_CONFIG = {
    # Operational Lines and their corresponding non-transfer stops
    'tram': { 'color': '#d1477a', 'linewidth': 1.5, 'alpha': 0.8, 'zorder': 4,
              'stop_color': '#333333', 'stop_size': 0.5 * 10, 'stop_marker': 'o', 'stop_alpha': 0.7, 'stop_zorder': 5 },
    'light_rail': { 'color': '#ff9900', 'linewidth': 2.0, 'alpha': 0.85, 'zorder': 3,
                    'stop_color': '#333333', 'stop_size': 0.5 * 10, 'stop_marker': 'o', 'stop_alpha': 0.7, 'stop_zorder': 4 },
    'metro': { 'color': '#1076e3', 'linewidth': 3.5, 'alpha': 0.8, 'zorder': 2,
               'stop_color': '#333333', 'stop_size': 1.5 * 10, 'stop_marker': 'o', 'stop_alpha': 0.8, 'stop_zorder': 3 },
    # 'metro': {'color': '#1079e3', 'linewidth': 3.5, 'alpha': 0.8, 'zorder': 2,
    #           'stop_color': '#333333', 'stop_size': 1.5 * 10, 'stop_marker': 'o', 'stop_alpha': 0.8, 'stop_zorder': 3},
    'train': { 'color': '#1adb1a', 'linewidth': 2.5, 'alpha': 0.8, 'zorder': 1,
               'stop_color': '#333333', 'stop_size': 1.2 * 10, 'stop_marker': 'o', 'stop_alpha': 0.7, 'stop_zorder': 2 },

    # Unified style for ALL transfer polygons
    'transfer': { 'facecolor': '#333333', 'edgecolor': 'none', 'linewidth': 0, 'alpha': 0.9, 'zorder': 10 },

    # Construction and Proposed lines use pale versions of operational colors.
    'construction_tram': { 'color': '#fce4ec', 'linewidth': 2.0, 'alpha': 1.0, 'zorder': 0 },
    'proposed_tram': { 'color': '#fce4ec', 'linewidth': 2.0, 'alpha': 1.0, 'zorder': 0 },

    'construction_light_rail': { 'color': '#fff8e1', 'linewidth': 2.0, 'alpha': 1.0, 'zorder': 0 },
    'proposed_light_rail': { 'color': '#fff8e1', 'linewidth': 2.0, 'alpha': 1.0, 'zorder': 0 },

    'construction_metro': { 'color': '#e3f2fd', 'linewidth': 3.5, 'alpha': 1.0, 'zorder': 0 },
    'proposed_metro': { 'color': '#e3f2fd', 'linewidth': 3.5, 'alpha': 1.0, 'zorder': 0 },

    'construction_train': { 'color': '#e8f5e9', 'linewidth': 2.0, 'alpha': 1.0, 'zorder': 0 },
    'proposed_train': { 'color': '#e8f5e9', 'linewidth': 2.0, 'alpha': 1.0, 'zorder': 0 },
}