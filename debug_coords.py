#!/usr/bin/env python3
"""
Debug Tool - Find and analyze elements near specific coordinates
"""

import requests
from osm_config import build_overpass_query, classify_stop, classify_route
from shapely.geometry import Point

# Overpass API endpoint
OVERPASS_URL = "http://overpass-api.de/api/interpreter"


def fetch_debug_data(center_lat, center_lon, radius_km=1.0):
    """Fetch all transit data around specific coordinates"""
    print(f"Fetching data around ({center_lat:.5f}, {center_lon:.5f}) within {radius_km}km...")

    # Fetch stops data
    query = build_overpass_query(center_lat, center_lon, radius_km, 'stops')

    response = requests.post(OVERPASS_URL, data={"data": query})
    if response.status_code != 200:
        raise RuntimeError(f"Overpass error: {response.status_code}")

    return response.json()


def debug_coordinate_area(target_lat, target_lon, search_radius_km=1.0, detail_radius_m=500):
    """
    Debug what's happening around specific coordinates

    Args:
        target_lat, target_lon: Coordinates to investigate
        search_radius_km: How far to search for data
        detail_radius_m: Show detailed info for elements within this distance
    """

    print(f"\n=== DEBUGGING COORDINATES ({target_lat:.5f}, {target_lon:.5f}) ===")

    # Fetch data
    data = fetch_debug_data(target_lat, target_lon, search_radius_km)

    elements = data.get('elements', [])
    print(f"Total elements fetched: {len(elements)}")

    # Target point for distance calculations
    target_point = Point(target_lon, target_lat)  # Note: Point takes (lon, lat)

    # Analyze all nodes
    nodes_nearby = []
    nodes_classified = []
    nodes_filtered = []

    for el in elements:
        if el['type'] == 'node':
            lon, lat = el['lon'], el['lat']
            element_point = Point(lon, lat)

            # Calculate distance in meters (approximate)
            distance_degrees = target_point.distance(element_point)
            distance_meters = distance_degrees * 111320  # Rough conversion

            # If within detail radius, analyze it
            if distance_meters <= detail_radius_m:
                nodes_nearby.append((el, distance_meters))

                # Test classification
                classification = classify_stop(el)

                name = el.get('tags', {}).get('name', 'Unnamed')
                tags = el.get('tags', {})

                print(f"\n--- Element at {distance_meters:.0f}m ---")
                print(f"Name: {name}")
                print(f"Coordinates: ({lat:.5f}, {lon:.5f})")
                print(f"Classification: {classification}")

                # Show relevant tags
                relevant_keys = ['railway', 'station', 'public_transport', 'train', 'tram',
                                 'subway', 'light_rail', 'bus', 'network:metro', 'operator']
                relevant_tags = {k: v for k, v in tags.items() if k in relevant_keys}
                if relevant_tags:
                    print(f"Relevant tags: {relevant_tags}")

                # Show ALL tags if it got classified but we don't understand why
                if classification and len(relevant_tags) < 2:
                    print(f"ALL TAGS: {tags}")

                if classification:
                    nodes_classified.append((el, classification, distance_meters))
                else:
                    nodes_filtered.append((el, distance_meters))

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Elements within {detail_radius_m}m: {len(nodes_nearby)}")
    print(f"Classified (will appear on map): {len(nodes_classified)}")
    print(f"Filtered out: {len(nodes_filtered)}")

    if nodes_classified:
        print(f"\n--- CLASSIFIED ELEMENTS ---")
        for el, classification, distance in nodes_classified:
            name = el.get('tags', {}).get('name', 'Unnamed')
            lat, lon = el['lat'], el['lon']
            print(f"  {classification}: {name} ({distance:.0f}m)")
            print(f"    Coordinates: {lat:.5f}, {lon:.5f}")
            print(f"    OSM ID: {el['id']}")

            # Show key classification tags
            tags = el.get('tags', {})
            key_tags = {}
            for key in ['railway', 'train', 'light_rail', 'station', 'operator', 'network']:
                if key in tags:
                    key_tags[key] = tags[key]
            print(f"    Key tags: {key_tags}")
            print()

    if nodes_filtered:
        print(f"\n--- FILTERED ELEMENTS ---")
        for el, distance in nodes_filtered[:5]:  # Show first 5
            name = el.get('tags', {}).get('name', 'Unnamed')
            railway = el.get('tags', {}).get('railway', '')
            bus = el.get('tags', {}).get('bus', '')
            print(f"  {name} ({distance:.0f}m) - railway:{railway}, bus:{bus}")

        if len(nodes_filtered) > 5:
            print(f"  ... and {len(nodes_filtered) - 5} more filtered elements")


def main():
    """Main debug function"""
    import sys

    if len(sys.argv) >= 3:
        # Command line: python debug_coords.py 52.49241 13.45817
        target_lat = float(sys.argv[1])
        target_lon = float(sys.argv[2])

        radius_km = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        detail_m = float(sys.argv[4]) if len(sys.argv) > 4 else 500

    else:
        # Your specific coordinates
        target_lat = 48.206595
        target_lon = 16.374650
        radius_km = 1.0
        detail_m = 500

        print("Usage: python debug_coords.py <lat> <lon> [search_radius_km] [detail_radius_m]")
        print(f"Using default coordinates: {target_lat}, {target_lon}")

    debug_coordinate_area(target_lat, target_lon, radius_km, detail_m)


if __name__ == "__main__":
    main()