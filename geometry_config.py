"""
Configuration for Geometric and Distance-Based Parameters
Used for stop processing and transfer visualization.
"""

# === TRANSFER DETECTION CONFIGURATION ===
# Distances in meters to detect a transfer between different stop types.
TRANSFER_MAX_DISTANCES = {
    'metro_tram': 200,
    'train_tram': 250,
    'train_metro': 300,
    'light_rail_tram': 200,
    'light_rail_metro': 250,
    'light_rail_train': 300,
    # Same-type connections are for merging nearby separate platforms of large stations
    'tram_tram': 1,
    'metro_metro': 250,
    'train_train': 250,
    'light_rail_light_rail': 250,
}

# === STOP PROCESSING CONFIGURATION ===
# Rules for merging or splitting stops during processing.
STOP_PROCESSING_CONFIG = {
    # Max distance (meters) between stops of the same name but different types
    # before they are forced to be separate entities rather than a single transfer point.
    'max_conflict_distance': 50,
}

# === TRANSFER VISUALIZATION CONFIGURATION ===
# Defines the geometry of the "stadium" shapes drawn for transfers.
TRANSFER_VISUAL_CONFIG = {
    # Extra length added to the distance between two stops to create the stadium rectangle.
    'stadium_length_padding': 80,
    # Width of the stadium shape.
    'stadium_width': 120,
}