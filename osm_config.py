"""
OSM Configuration for Transit Map Generator - Final Robust Version
"""
import math

def create_bbox_from_center(center_lat, center_lon, radius_km):
    """
    Create a bounding box around center coordinates, robust for all hemispheres.
    """
    if radius_km < 0: radius_km = 0
    lat_offset = radius_km / 111.0
    lon_offset = radius_km / (111.0 * math.cos(math.radians(center_lat)))
    south, north = center_lat - lat_offset, center_lat + lat_offset
    west, east = center_lon - lon_offset, center_lon + lon_offset
    return min(south, north), min(west, east), max(south, north), max(west, east)

def build_overpass_query(center_lat, center_lon, radius_km, query_type):
    """
    Builds Overpass queries using a global [bbox:...] setting to bypass a server-side
    parser bug with negative coordinates in the (bbox:...) filter.
    """
    south, west, north, east = create_bbox_from_center(center_lat, center_lon, radius_km)
    bbox_str = f"{south:.6f},{west:.6f},{north:.6f},{east:.6f}"

    # We now define the bbox once at the top. This is the fix.
    base_query_settings = f"[out:json][bbox:{bbox_str}];"

    if query_type == 'tram_and_light_rail':
        return f"""
        {base_query_settings}
        (
          relation["type"="route"]["route"="tram"]["state"!="proposed"];
          relation["type"="route"]["route"="light_rail"]["state"!="proposed"][!"network:metro"];
        );
        out body;
        >;
        out skel qt;
        """
    elif query_type == 'metro':
        return f"""
        {base_query_settings}
        (
          relation["type"="route"]["route"~"^(subway|metro)$"];
        );
        out body;
        >;
        out skel qt;
        """
    elif query_type == 'train':
        return f"""
        {base_query_settings}
        (
          relation["type"="route"]["route"="train"]["service"!~"^(international|long_distance)$"];
          relation["type"="route"]["route"="monorail"];
          relation["type"="route"]["route"="light_rail"]["network:metro"];
        );
        out body;
        >;
        out skel qt;
        """
    elif query_type == 'stops':
        return f"""
        {base_query_settings}
        (
          node["railway"~"^(tram_stop|station|stop|halt)$"];
          node["public_transport"~"^(station|stop_position)$"];
          node["train"="yes"];
          node["tram"="yes"];
          node["subway"="yes"];
          node["light_rail"="yes"];
        );
        out body;
        """
    else:
        raise ValueError(f"Unknown query type: {query_type}")

def classify_route(route_element):
    tags = route_element.get('tags', {}); route = tags.get('route', '')
    if route == 'tram': return 'tram'
    if route == 'light_rail': return 'train' if tags.get('network:metro') else 'light_rail'
    if route in ['subway', 'metro']: return 'metro'
    if route in ['train', 'monorail']: return 'train'
    return None

def classify_stop(stop_element):
    tags = stop_element.get('tags', {});
    if tags.get('railway') in ['proposed', 'construction'] or any(k.startswith('abandoned:') for k in tags): return None
    if tags.get('subway') == 'yes' or tags.get('station') == 'subway': return 'metro'
    if tags.get('tram') == 'yes' or tags.get('railway') == 'tram_stop': return 'tram'
    if tags.get('light_rail') == 'yes' or tags.get('station') == 'light_rail': return 'light_rail'
    if tags.get('train') == 'yes': return 'train'
    if tags.get('railway') in ['station', 'stop', 'halt']: return 'train'
    return None

VISUAL_CONFIG = {
    'tram': { 'color': '#d1477a', 'linewidth': 1.5, 'alpha': 0.8, 'zorder': 4, 'stop_color': '#a63861', 'stop_size': 8, 'stop_marker': 'o' },
    'light_rail': { 'color': '#ff9900', 'linewidth': 2.0, 'alpha': 0.85, 'zorder': 3, 'stop_color': '#cc6600', 'stop_size': 15, 'stop_marker': 'd' },
    'metro': { 'color': '#1079e3', 'linewidth': 3.5, 'alpha': 0.9, 'zorder': 2, 'stop_color': '#0053a6', 'stop_size': 30, 'stop_marker': 'o' },
    'train': { 'color': '#1adb1a', 'linewidth': 2.5, 'alpha': 0.8, 'zorder': 1, 'stop_color': '#228B22', 'stop_size': 25, 'stop_marker': 's' },
    'transfer': { 'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 2, 'alpha': 0.9, 'zorder': 10 }
}