"""
OSM Configuration for Transit Map Generator - Clean & Simple Approach
Comprehensive fetching with clear classification rules
"""

def create_bbox_from_center(center_lat, center_lon, radius_km):
    """Create a bounding box around center coordinates"""
    import math

    lat_offset = radius_km / 111.0
    lon_offset = radius_km / (111.0 * math.cos(math.radians(center_lat)))

    south = center_lat - lat_offset
    north = center_lat + lat_offset
    west = center_lon - lon_offset
    east = center_lon + lon_offset

    return south, west, north, east


def build_overpass_query(center_lat, center_lon, radius_km, query_type):
    """
    Build comprehensive Overpass queries - fetch everything, classify later
    """
    bbox = create_bbox_from_center(center_lat, center_lon, radius_km)
    south, west, north, east = bbox
    bbox_str = f"{south},{west},{north},{east}"

    if query_type == 'tram':
        return f"""
        [out:json];
        (
          relation["type"="route"]["route"="tram"]["state"!="proposed"](bbox:{bbox_str});
          relation["type"="route"]["route"="light_rail"]["state"!="proposed"][!"network:metro"](bbox:{bbox_str});
        );
        out body;
        >;
        out skel qt;
        """

    elif query_type == 'metro':
        return f"""
        [out:json];
        (
          relation["type"="route"]["route"="subway"](bbox:{bbox_str});
          relation["type"="route"]["route"="metro"](bbox:{bbox_str});
        );
        out body;
        >;
        out skel qt;
        """

    elif query_type == 'train':
        return f"""
        [out:json];
        (
          relation["type"="route"]["route"="train"]["service"!="international"]["service"!="long_distance"](bbox:{bbox_str});
          relation["type"="route"]["route"="light_rail"]["network:metro"](bbox:{bbox_str});
          relation["type"="route"]["route"="monorail"](bbox:{bbox_str});
        );
        out body;
        >;
        out skel qt;
        """

    elif query_type == 'stops':
        return f"""
        [out:json];
        (
          // All tram infrastructure
          node["railway"="tram_stop"](bbox:{bbox_str});
          node["tram"="yes"](bbox:{bbox_str});
          
          // All railway stations and stops
          node["railway"="station"](bbox:{bbox_str});
          node["railway"="stop"](bbox:{bbox_str});
          node["railway"="halt"](bbox:{bbox_str});
          
          // All explicit train/metro infrastructure
          node["train"="yes"](bbox:{bbox_str});
          node["subway"="yes"](bbox:{bbox_str});
          
          // Public transport stations (covers edge cases)
          node["public_transport"~"^(station|stop_position)$"](bbox:{bbox_str});
        );
        out body;
        """

    else:
        raise ValueError(f"Unknown query type: {query_type}")


def classify_stop(stop_element):
    """
    Conservative, evidence-based stop classification
    Requires strong transit infrastructure evidence to prevent phantom stops
    """
    tags = stop_element.get('tags', {})
    name = tags.get('name', 'Unnamed')

    # === EARLY FILTERS: Remove non-operational infrastructure ===

    railway = tags.get('railway', '')
    if railway in ['proposed', 'construction']:
        return None

    if tags.get('railway:preserved') == 'yes':
        return None

    # Filter abandoned infrastructure
    if any(key.startswith('abandoned:') for key in tags.keys()):
        return None

    # Filter technical/non-passenger stations
    if railway in ['technical_station', 'service_station', 'yard']:
        return None

    # === EARLY FILTER: Exclude bus/boat-only stops ===

    has_bus = tags.get('bus') == 'yes' or tags.get('coach') == 'yes'
    has_boat = tags.get('ferry') == 'yes' or tags.get('boat') == 'yes'
    has_tram = tags.get('tram') == 'yes'

    if (has_bus or has_boat) and not has_tram and 'railway' not in tags:
        return None

    # === EXTRACT KEY INDICATORS ===

    # Railway infrastructure
    railway = tags.get('railway', '')
    station = tags.get('station', '')

    # Transit type flags
    has_train = tags.get('train') == 'yes'
    has_subway = tags.get('subway') == 'yes'
    has_light_rail = tags.get('light_rail') == 'yes'
    has_network_metro = bool(tags.get('network:metro'))

    # Network identifiers
    operator = tags.get('operator', '').lower()
    network = tags.get('network', '').lower()

    # === PRIORITY 1: Strong Type Evidence ===

    # Tram infrastructure
    if railway == 'tram_stop' or has_tram:
        return 'tram'

    # Metro/Subway infrastructure
    if (railway == 'station' and station == 'subway') or has_subway:
        return 'metro'

    # Explicit train infrastructure
    if has_train:
        return 'train'

    # === PRIORITY 2: Light Rail with Context ===

    if has_light_rail or station == 'light_rail':
        # Light rail + metro network = S-Bahn (train)
        # Light rail alone = tram
        return 'train' if has_network_metro else 'tram'

    # === PRIORITY 3: Railway Infrastructure (Conservative) ===

    # Must have railway tag + supporting evidence
    if railway in ['station', 'stop', 'halt']:

        # Look for supporting transit evidence
        transit_evidence = [
            'train' in tags,
            'light_rail' in tags,
            'subway' in tags,
            has_network_metro,
            's-bahn' in operator,
            's-bahn' in network,
            any(keyword in operator for keyword in ['db', 'bahn', 'rail']),
            any(keyword in network for keyword in ['db', 'bahn', 'rail'])
        ]

        if any(transit_evidence):
            # Determine type based on evidence
            if station == 'subway' or has_subway:
                return 'metro'
            elif 'tram' in operator or 'tram' in network:
                return 'tram'
            else:
                return 'train'  # Default for rail infrastructure

    # === NO WEAK FALLBACKS ===
    # Removed: public_transport alone is not sufficient evidence
    # This prevents "BVG Service" shops and similar false positives

    # === DEBUG: Log potential misses ===

    # Only log if it has multiple transit indicators (might be legitimate)
    transit_indicators = sum([
        bool(railway),
        bool(station),
        has_train,
        has_tram,
        has_subway,
        has_light_rail,
        has_network_metro,
        'transport' in tags.get('public_transport', ''),
        's-bahn' in operator,
        's-bahn' in network
    ])

    if transit_indicators >= 2:
        relevant_tags = {k: v for k, v in tags.items()
                         if k in ['railway', 'station', 'public_transport', 'train', 'tram',
                                  'subway', 'light_rail', 'network:metro', 'operator', 'network']}
        print(f"UNCLASSIFIED (potential miss): {name} | Tags: {relevant_tags}")

    return None

def classify_route(route_element):
    """
    Clear route classification with special cases
    """
    tags = route_element.get('tags', {})
    route_type = tags.get('route', '')
    ref = tags.get('ref', 'No-ref')

    # Boolean flags
    has_network_metro = bool(tags.get('network:metro'))

    # String values
    operator = tags.get('operator', '').lower()
    service = tags.get('service', '')

    # === Special operator cases ===

    if 'máv-hév' in operator:  # Budapest suburban rail
        return 'train'

    # === Standard route classification ===

    if route_type == 'tram':
        return 'tram'

    elif route_type == 'subway' or route_type == 'metro':
        return 'metro'

    elif route_type == 'light_rail':
        # Light rail with metro network = train (S-Bahn)
        # Light rail without metro network = tram
        return 'train' if has_network_metro else 'tram'

    elif route_type == 'train':
        # Exclude long-distance services (already filtered in query)
        return 'train'

    elif route_type == 'monorail':
        return 'train'  # Treat monorails as train-like

    # === DEBUG: Log unclassified routes ===

    if route_type:  # Only log if it has a route type
        relevant_tags = {k: v for k, v in tags.items()
                        if k in ['route', 'network:metro', 'operator', 'ref', 'service']}
        print(f"UNCLASSIFIED ROUTE: {ref} | Tags: {relevant_tags}")

    return None


# === VISUAL CONFIGURATION ===

VISUAL_CONFIG = {
    'tram': {
        'color': '#d1477a',           # Lighter pink for lines
        'linewidth': 1.5,
        'alpha': 0.8,
        'zorder': 3,
        'stop_color': '#a63861',      # Darker pink for stops
        'stop_size': 8,
        'stop_marker': 'o'
    },
    'metro': {
        'color': '#1079e3',           # Lighter blue for lines
        'linewidth': 3.5,
        'alpha': 0.9,
        'zorder': 2,
        'stop_color': '#0053a6',      # Darker blue for stops
        'stop_size': 30,
        'stop_marker': 'o'
    },
    'train': {
        'color': '#1adb1a',           # Lighter green for lines
        'linewidth': 2.5,
        'alpha': 0.8,
        'zorder': 1,
        'stop_color': '#228B22',      # Darker green for stops
        'stop_size': 25,
        'stop_marker': 's'  # Square
    },
    'transfer': {
        'facecolor': 'white',
        'edgecolor': 'black',
        'linewidth': 2,
        'alpha': 0.9,
        'zorder': 8
    }
}

# === TRANSFER DETECTION CONFIGURATION ===

TRANSFER_CONFIG = {
    'metro_tram': {
        'max_distance': 200,  # meters
        'description': 'Metro-Tram interchange'
    },
    'train_tram': {
        'max_distance': 250,  # meters
        'description': 'Train-Tram interchange'
    },
    'train_metro': {
        'max_distance': 300,  # meters
        'description': 'Train-Metro interchange'
    },
    'tram_tram': {
        'max_distance': 1,  # meters
        'description': 'Close tram stops connection'
    },
    'metro_metro': {
        'max_distance': 250,  # meters
        'description': 'Close metro stations connection'
    },
    'train_train': {
        'max_distance': 250,  # meters
        'description': 'Close train stations connection'
    }
}

# === DEBUGGING UTILITIES ===

def debug_classification_stats(all_elements):
    """
    Print classification statistics for debugging
    """
    stats = {'tram': 0, 'metro': 0, 'train': 0, 'unclassified': 0}

    for element in all_elements:
        if element.get('type') == 'node':
            classification = classify_stop(element)
            if classification:
                stats[classification] += 1
            else:
                stats['unclassified'] += 1
        elif element.get('type') == 'relation':
            classification = classify_route(element)
            if classification:
                stats[classification] += 1
            else:
                stats['unclassified'] += 1

    print("\n=== Classification Stats ===")
    for transit_type, count in stats.items():
        print(f"{transit_type}: {count}")
    print("=" * 30)

    return stats