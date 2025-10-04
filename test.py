"""
Debug script to analyze what stops are being detected near a specific location
"""
import requests
import math
from pyproj import Transformer
from shapely.geometry import Point

OVERPASS_URL = "http://overpass-api.de/api/interpreter"


def create_bbox_from_center(center_lat, center_lon, radius_km):
    """Create a bounding box around center coordinates."""
    if radius_km < 0: radius_km = 0
    lat_offset = radius_km / 111.0
    lon_offset = radius_km / (111.0 * math.cos(math.radians(center_lat)))
    south, north = center_lat - lat_offset, center_lat + lat_offset
    west, east = center_lon - lon_offset, center_lon + lon_offset
    return min(south, north), min(west, east), max(south, north), max(west, east)


def fetch_stops_debug(center_lat, center_lon, radius_km=0.5):
    """Fetch stops with debug info."""
    south, west, north, east = create_bbox_from_center(center_lat, center_lon, radius_km)
    bbox_str = f"{south:.6f},{west:.6f},{north:.6f},{east:.6f}"

    query = f"""[out:json][timeout:60][bbox:{bbox_str}];
    (node["railway"~"^(tram_stop|station|stop|halt)$"];
     node["public_transport"~"^(station|stop_position)$"];
     node["train"="yes"]; node["tram"="yes"]; node["subway"="yes"]; node["light_rail"="yes"];);
    out body;"""

    response = requests.post(OVERPASS_URL, data={"data": query})
    if response.status_code == 200:
        return response.json()
    else:
        raise RuntimeError(f"Overpass error: {response.status_code}")


def classify_stop(stop_element):
    """Same classification as main script."""
    tags = stop_element.get('tags', {})
    if tags.get('subway') == 'yes' or tags.get('station') == 'subway':
        return 'metro'
    if tags.get('tram') == 'yes' or tags.get('railway') == 'tram_stop':
        return 'tram'
    if tags.get('light_rail') == 'yes' or tags.get('station') == 'light_rail':
        return 'light_rail'
    if tags.get('train') == 'yes':
        return 'train'
    if tags.get('railway') in ['station', 'stop', 'halt']:
        return 'train'
    return None


def debug_stops_near_location(target_lat, target_lon, search_radius_km=0.5, max_distance_m=200):
    """
    Debug stops near a specific location.

    Args:
        target_lat, target_lon: The location to search around
        search_radius_km: How large an area to search
        max_distance_m: Only show stops within this distance from target
    """
    print(f"\n=== Debugging stops near {target_lat}, {target_lon} ===")
    print(f"Search radius: {search_radius_km} km")
    print(f"Max distance filter: {max_distance_m} m\n")

    # Fetch data
    data = fetch_stops_debug(target_lat, target_lon, search_radius_km)

    # Create transformer for Vienna (UTM Zone 33N)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    target_coord = transformer.transform(target_lon, target_lat)
    target_point = Point(target_coord)

    print(f"Found {len(data['elements'])} total stop elements in search area\n")

    # Analyze each stop
    stops_by_type = {'metro': [], 'tram': [], 'light_rail': [], 'train': [], 'unclassified': []}

    for el in data['elements']:
        if el['type'] == 'node' and 'tags' in el:
            coord = transformer.transform(el['lon'], el['lat'])
            distance = target_point.distance(Point(coord))

            if distance <= max_distance_m:
                stop_type = classify_stop(el)
                category = stop_type if stop_type else 'unclassified'

                stops_by_type[category].append({
                    'id': el['id'],
                    'name': el['tags'].get('name', 'Unnamed'),
                    'distance': distance,
                    'tags': el['tags'],
                    'lat': el['lat'],
                    'lon': el['lon']
                })

    # Print results
    for category in ['metro', 'tram', 'light_rail', 'train', 'unclassified']:
        stops = stops_by_type[category]
        if stops:
            print(f"\n{'=' * 70}")
            print(f"{category.upper()} STOPS ({len(stops)} found):")
            print('=' * 70)

            # Sort by distance
            stops.sort(key=lambda x: x['distance'])

            for stop in stops:
                print(f"\nNode ID: {stop['id']}")
                print(f"Name: {stop['name']}")
                print(f"Distance: {stop['distance']:.1f} m")
                print(f"Location: {stop['lat']:.6f}, {stop['lon']:.6f}")
                print(f"OSM Link: https://www.openstreetmap.org/node/{stop['id']}")
                print("Tags:")
                for key, value in sorted(stop['tags'].items()):
                    print(f"  {key} = {value}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY:")
    print('=' * 70)
    total = sum(len(stops_by_type[cat]) for cat in stops_by_type)
    print(f"Total stops within {max_distance_m}m: {total}")
    for category in ['metro', 'tram', 'light_rail', 'train', 'unclassified']:
        count = len(stops_by_type[category])
        if count > 0:
            print(f"  {category}: {count}")


if __name__ == "__main__":
    # The location you mentioned (Stephansplatz area in Vienna)
    target_lat = 48.197552
    target_lon = 16.348042

    debug_stops_near_location(target_lat, target_lon, search_radius_km=0.5, max_distance_m=200)