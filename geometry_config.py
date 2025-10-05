"""
Configuration for Geometric and Distance-Based Parameters
Used for stop processing and transfer visualization.
"""

# === TRANSFER DETECTION CONFIGURATION ===
# Distances in meters to detect a transfer between different stop types.
# TRANSFER_MAX_DISTANCES = {
#     'metro_tram': 200,
#     'train_tram': 250,
#     'train_metro': 300,
#     'light_rail_tram': 10,
#     'light_rail_metro': 250,
#     'light_rail_train': 300,
#     'tram_tram': 10,
#     'metro_metro': 250,
#     'train_train': 250,
#     'light_rail_light_rail': 10,
# }

TRANSFER_MAX_DISTANCES = {
    'metro_tram': 10,
    'train_tram': 10,
    'train_metro': 10,
    'light_rail_tram': 10,
    'light_rail_metro': 10,
    'light_rail_train': 10,
    'tram_tram': 10,
    'metro_metro': 10,
    'train_train': 10,
    'light_rail_light_rail': 10,
}

# === STOP PROCESSING CONFIGURATION ===
# Rules for merging or splitting stops during processing.
STOP_PROCESSING_CONFIG = {
    'max_conflict_distance': 50,
    'stop_proximity_buffer': 50,
    'line_simplification_tolerance': 10,
}

# === TRANSFER VISUALIZATION CONFIGURATION ===
# Defines the geometry of the "stadium" shapes drawn for transfers.
# TRANSFER_VISUAL_CONFIG = {
#     'stadium_length_padding': 80,
#     'stadium_width': 120, # This defines the diameter of single-point circles (radius is half)
# }

TRANSFER_VISUAL_CONFIG = {
    'stadium_length_padding': 20,
    'stadium_width': 90, # This defines the diameter of single-point circles (radius is half)
}