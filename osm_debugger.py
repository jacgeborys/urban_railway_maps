#!/usr/bin/env python3
"""
OSM Railway Data Debugger
Analyzes all railway routes and stops in a city to understand the data structure
and develop better classification strategies
"""

import requests
import json
from collections import defaultdict, Counter
from cities import CITIES

OVERPASS_URL = "http://overpass-api.de/api/interpreter"


def fetch_all_railway_routes(city_name):
    """Fetch ALL railway routes in a city for analysis"""
    print(f"Fetching all railway routes for {city_name}...")

    query = f"""
    [out:json];
    area["name"="{city_name}"]->.city;
    relation["type"="route"]["route"~"^(tram|light_rail|subway|train|bus|trolleybus|ferry)$"](area.city);
    out tags;
    """

    response = requests.post(OVERPASS_URL, data={"data": query})
    if response.status_code != 200:
        raise RuntimeError(f"Overpass error: {response.status_code}")

    return response.json()


def fetch_all_railway_stops(city_name):
    """Fetch ALL railway-related stops/stations in a city"""
    print(f"Fetching all railway stops for {city_name}...")

    query = f"""
    [out:json];
    area["name"="{city_name}"]->.city;
    node["railway"~"^(station|stop|halt|tram_stop)$"](area.city);
    out tags;
    """

    response = requests.post(OVERPASS_URL, data={"data": query})
    if response.status_code != 200:
        raise RuntimeError(f"Overpass error: {response.status_code}")

    return response.json()


def analyze_routes(routes_data, city_name):
    """Analyze all route types and their properties"""
    print(f"\n=== ROUTE ANALYSIS for {city_name} ===")

    route_types = Counter()
    route_details = defaultdict(list)

    for element in routes_data["elements"]:
        if element["type"] == "relation":
            tags = element.get("tags", {})
            route_type = tags.get("route", "unknown")

            route_types[route_type] += 1

            # Collect detailed info for each route
            route_info = {
                "ref": tags.get("ref", "no_ref"),
                "name": tags.get("name", "no_name"),
                "operator": tags.get("operator", "no_operator"),
                "network": tags.get("network", "no_network"),
                "network:metro": tags.get("network:metro", "no_network_metro"),
                "service": tags.get("service", "no_service"),
                "state": tags.get("state", "no_state"),
                "public_transport": tags.get("public_transport", "no_pt"),
                "all_tags": dict(tags)
            }

            route_details[route_type].append(route_info)

    # Print summary
    print(f"Route type counts:")
    for route_type, count in route_types.most_common():
        print(f"  {route_type}: {count}")

    # Detailed analysis for transit routes
    transit_types = ["tram", "light_rail", "subway", "train"]

    for route_type in transit_types:
        if route_type in route_details:
            print(f"\n--- {route_type.upper()} ROUTES ---")
            routes = route_details[route_type]

            # Analyze common properties
            operators = Counter(r["operator"] for r in routes)
            networks = Counter(r["network"] for r in routes)
            network_metros = Counter(r["network:metro"] for r in routes)
            services = Counter(r["service"] for r in routes)

            print(f"  Count: {len(routes)}")
            print(f"  Operators: {dict(operators.most_common(5))}")
            print(f"  Networks: {dict(networks.most_common(5))}")
            print(f"  Network:metro: {dict(network_metros.most_common(5))}")
            print(f"  Services: {dict(services.most_common(5))}")

            # Show some examples
            print(f"  Example routes:")
            for i, route in enumerate(routes[:3]):
                print(f"    {i + 1}. {route['ref']} - {route['name'][:50]}")
                print(f"       operator: {route['operator']}")
                print(f"       network: {route['network']}")
                print(f"       network:metro: {route['network:metro']}")
                print(f"       service: {route['service']}")

    return route_details


def analyze_stops(stops_data, city_name):
    """Analyze all stop/station types and their properties"""
    print(f"\n=== STOP ANALYSIS for {city_name} ===")

    railway_types = Counter()
    station_types = Counter()
    stop_details = defaultdict(list)

    for element in stops_data["elements"]:
        if element["type"] == "node":
            tags = element.get("tags", {})
            railway = tags.get("railway", "unknown")
            station = tags.get("station", "no_station")

            railway_types[railway] += 1

            if station != "no_station":
                station_types[station] += 1

            # Collect detailed info
            stop_info = {
                "name": tags.get("name", "no_name"),
                "railway": railway,
                "station": station,
                "operator": tags.get("operator", "no_operator"),
                "network": tags.get("network", "no_network"),
                "network:metro": tags.get("network:metro", "no_network_metro"),
                "public_transport": tags.get("public_transport", "no_pt"),
                "train": tags.get("train", "no_train"),
                "subway": tags.get("subway", "no_subway"),
                "tram": tags.get("tram", "no_tram"),
                "light_rail": tags.get("light_rail", "no_light_rail"),
                "uic_ref": tags.get("uic_ref", "no_uic"),
                "all_tags": dict(tags)
            }

            stop_details[railway].append(stop_info)

    # Print summary
    print(f"Railway type counts:")
    for railway_type, count in railway_types.most_common():
        print(f"  {railway_type}: {count}")

    print(f"\nStation type counts:")
    for station_type, count in station_types.most_common():
        print(f"  {station_type}: {count}")

    # Detailed analysis for each railway type
    for railway_type, stops in stop_details.items():
        if len(stops) > 0:
            print(f"\n--- {railway_type.upper()} STOPS ---")

            # Analyze properties
            operators = Counter(s["operator"] for s in stops)
            networks = Counter(s["network"] for s in stops)
            network_metros = Counter(s["network:metro"] for s in stops)
            stations = Counter(s["station"] for s in stops)

            print(f"  Count: {len(stops)}")
            print(f"  Station types: {dict(stations.most_common())}")
            print(f"  Operators: {dict(operators.most_common(3))}")
            print(f"  Networks: {dict(networks.most_common(3))}")
            print(f"  Network:metro: {dict(network_metros.most_common(3))}")

            # Show examples
            print(f"  Example stops:")
            for i, stop in enumerate(stops[:3]):
                print(f"    {i + 1}. {stop['name'][:40]}")
                print(f"       station: {stop['station']}")
                print(f"       operator: {stop['operator']}")
                print(f"       network:metro: {stop['network:metro']}")
                print(f"       train: {stop['train']}, tram: {stop['tram']}, light_rail: {stop['light_rail']}")

    return stop_details


def analyze_light_rail_confusion(routes_data, stops_data, city_name):
    """Special analysis for light rail classification issues"""
    print(f"\n=== LIGHT RAIL CONFUSION ANALYSIS for {city_name} ===")

    # Analyze light_rail routes
    light_rail_routes = []
    for element in routes_data["elements"]:
        if element["type"] == "relation":
            tags = element.get("tags", {})
            if tags.get("route") == "light_rail":
                light_rail_routes.append(tags)

    if light_rail_routes:
        print(f"Found {len(light_rail_routes)} light_rail routes:")
        for i, route in enumerate(light_rail_routes):
            print(f"  {i + 1}. {route.get('ref', 'no_ref')} - {route.get('name', 'no_name')[:50]}")
            print(f"     operator: {route.get('operator', 'no_operator')}")
            print(f"     network: {route.get('network', 'no_network')}")
            print(f"     network:metro: {route.get('network:metro', 'no_network_metro')}")
            print(f"     service: {route.get('service', 'no_service')}")

    # Analyze light_rail stops
    light_rail_stops = []
    for element in stops_data["elements"]:
        if element["type"] == "node":
            tags = element.get("tags", {})
            if (tags.get("station") == "light_rail" or
                    tags.get("light_rail") == "yes"):
                light_rail_stops.append(tags)

    if light_rail_stops:
        print(f"\nFound {len(light_rail_stops)} light_rail stops:")
        for i, stop in enumerate(light_rail_stops[:5]):  # Show first 5
            print(f"  {i + 1}. {stop.get('name', 'no_name')[:40]}")
            print(f"     railway: {stop.get('railway', 'no_railway')}")
            print(f"     station: {stop.get('station', 'no_station')}")
            print(f"     operator: {stop.get('operator', 'no_operator')}")
            print(f"     network:metro: {stop.get('network:metro', 'no_network_metro')}")
            print(f"     train: {stop.get('train', 'no_train')}")


def generate_classification_suggestions(routes_details, stops_details, city_name):
    """Generate classification suggestions based on the analysis"""
    print(f"\n=== CLASSIFICATION SUGGESTIONS for {city_name} ===")

    suggestions = []

    # Analyze operators and networks to suggest classification rules
    if "light_rail" in routes_details:
        light_rail_routes = routes_details["light_rail"]

        # Group by network:metro presence
        with_metro = [r for r in light_rail_routes if r["network:metro"] != "no_network_metro"]
        without_metro = [r for r in light_rail_routes if r["network:metro"] == "no_network_metro"]

        if with_metro and without_metro:
            suggestions.append("MIXED light_rail: Some have network:metro, some don't")
            suggestions.append(f"  - With network:metro ({len(with_metro)}): likely TRAIN/S-Bahn")
            suggestions.append(f"  - Without network:metro ({len(without_metro)}): likely TRAM")
        elif with_metro:
            suggestions.append("ALL light_rail routes have network:metro -> classify as TRAIN")
        elif without_metro:
            suggestions.append("NO light_rail routes have network:metro -> classify as TRAM")

    # Analyze operators
    all_operators = set()
    for route_type, routes in routes_details.items():
        for route in routes:
            if route["operator"] != "no_operator":
                all_operators.add((route_type, route["operator"]))

    if all_operators:
        suggestions.append("\nOperator-based classification suggestions:")
        operator_groups = defaultdict(list)
        for route_type, operator in all_operators:
            operator_groups[operator].append(route_type)

        for operator, types in operator_groups.items():
            suggestions.append(f"  - {operator}: {set(types)}")

    for suggestion in suggestions:
        print(suggestion)

    return suggestions


def debug_city(city_key):
    """Complete debugging analysis for a city"""
    if city_key not in CITIES:
        print(f"City '{city_key}' not found. Available cities: {list(CITIES.keys())}")
        return

    city_config = CITIES[city_key]
    city_name = city_config['name']

    try:
        # Fetch data
        routes_data = fetch_all_railway_routes(city_name)
        stops_data = fetch_all_railway_stops(city_name)

        # Analyze
        routes_details = analyze_routes(routes_data, city_name)
        stops_details = analyze_stops(stops_data, city_name)
        analyze_light_rail_confusion(routes_data, stops_data, city_name)
        generate_classification_suggestions(routes_details, stops_details, city_name)

        # Save detailed data for further analysis
        debug_data = {
            "city": city_name,
            "routes": routes_details,
            "stops": stops_details,
            "raw_routes": routes_data,
            "raw_stops": stops_data
        }

        filename = f"debug_{city_key}_railway_data.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed data saved to {filename}")

    except Exception as e:
        print(f"Error debugging {city_name}: {e}")


def compare_cities(city_keys):
    """Compare railway data across multiple cities"""
    print(f"\n=== MULTI-CITY COMPARISON ===")

    city_data = {}

    for city_key in city_keys:
        if city_key not in CITIES:
            print(f"Skipping unknown city: {city_key}")
            continue

        city_config = CITIES[city_key]
        city_name = city_config['name']

        try:
            routes_data = fetch_all_railway_routes(city_name)
            stops_data = fetch_all_railway_stops(city_name)

            # Quick analysis
            route_counts = Counter()
            stop_counts = Counter()

            for element in routes_data["elements"]:
                if element["type"] == "relation":
                    route_type = element.get("tags", {}).get("route", "unknown")
                    route_counts[route_type] += 1

            for element in stops_data["elements"]:
                if element["type"] == "node":
                    railway_type = element.get("tags", {}).get("railway", "unknown")
                    stop_counts[railway_type] += 1

            city_data[city_name] = {
                "routes": dict(route_counts),
                "stops": dict(stop_counts)
            }

        except Exception as e:
            print(f"Error processing {city_name}: {e}")

    # Print comparison
    print(f"\nRoute type comparison:")
    all_route_types = set()
    for data in city_data.values():
        all_route_types.update(data["routes"].keys())

    for route_type in sorted(all_route_types):
        print(f"\n{route_type}:")
        for city, data in city_data.items():
            count = data["routes"].get(route_type, 0)
            print(f"  {city}: {count}")

    print(f"\nStop type comparison:")
    all_stop_types = set()
    for data in city_data.values():
        all_stop_types.update(data["stops"].keys())

    for stop_type in sorted(all_stop_types):
        print(f"\n{stop_type}:")
        for city, data in city_data.items():
            count = data["stops"].get(stop_type, 0)
            print(f"  {city}: {count}")


def main():
    """Main debugging function"""
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "compare" and len(sys.argv) > 2:
            # Compare multiple cities
            city_keys = [city.lower() for city in sys.argv[2:]]
            compare_cities(city_keys)
        else:
            # Debug single city
            debug_city(command)
    else:
        # Interactive mode
        print("OSM Railway Data Debugger")
        print("\nAvailable cities:")
        for key, config in CITIES.items():
            print(f"  {key}: {config['name']}")

        print("\nCommands:")
        print("  python osm_debugger.py <city_key>")
        print("  python osm_debugger.py compare <city1> <city2> ...")

        city_key = input("\nEnter city key to debug: ").lower().strip()
        if city_key:
            debug_city(city_key)


if __name__ == "__main__":
    main()