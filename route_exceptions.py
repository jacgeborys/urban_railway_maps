"""
Route Exceptions Configuration
Handles special cases where OSM tagging doesn't match actual transit type
or where routes should be excluded entirely.
"""

# Relations to completely exclude from the map (e.g., proposed routes without proper tags)
EXCLUDED_RELATIONS = [
    16376023,  # Planowana trasa tramwajowa do Dworca Zachodniego (Warsaw) - proposed tram line
    19659733,
    3094380,
    19427410,
    19400819,
    19230796,
    11591890,
    17064124,
    17064125
]

# Relations that should be reclassified to a different type
# Format: {relation_id: 'target_type'}
# Valid target types: 'tram', 'light_rail', 'metro', 'train'
RECLASSIFIED_RELATIONS = {
    8780893: 'train',  # PKM 3: Poznań Główny => Wągrowiec (heavy rail misclassified as light_rail)
    8780894: 'train',  # PKM 4: (heavy rail misclassified as light_rail)
    19249613: 'light_rail',  # In Rome
    19249612: 'light_rail',  # In Rome
    1721540: 'light_rail',  # In Rome
    1721539: 'light_rail',  # In Rome
}

def get_route_exception(relation_id):
    """
    Check if a relation has an exception rule.

    Returns:
        - None if the relation should be excluded
        - A string (transit type) if it should be reclassified
        - False if no exception applies
    """
    if relation_id in EXCLUDED_RELATIONS:
        return None

    if relation_id in RECLASSIFIED_RELATIONS:
        return RECLASSIFIED_RELATIONS[relation_id]

    return False