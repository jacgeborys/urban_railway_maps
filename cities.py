# Configuration for different cities
# NOTE: 'name' is now only used for display purposes, not for OSM queries
# OSM queries use coordinates with bounding box instead

CITIES = {
    'warszawa': {
        'name': 'Warsaw',
        'center': (52.2297, 21.0122),
        'bounds_km': 20,
        'utm_zone': 34  # UTM Zone 34N for Poland
    },
    'gdansk': {
        'name': 'Gdańsk',
        'center': (54.3520, 18.6466),
        'bounds_km': 20,
        'utm_zone': 33  # UTM Zone 33N
    },
    'krakow': {
        'name': 'Kraków',
        'center': (50.0647, 19.9450),
        'bounds_km': 20,
        'utm_zone': 34  # UTM Zone 34N
    },
    'berlin': {
        'name': 'Berlin',
        'center': (52.5200, 13.4050),
        'bounds_km': 20,
        'utm_zone': 33  # UTM Zone 33N
    },
    'poznan': {
        'name': 'Poznań',
        'center': (52.4064, 16.9252),
        'bounds_km': 20,
        'utm_zone': 33  # UTM Zone 33N
    },
    'wroclaw': {
        'name': 'Wrocław',
        'center': (51.1079, 17.0385),
        'bounds_km': 20,
        'utm_zone': 33  # UTM Zone 33N
    },
    'vienna': {
        'name': 'Vienna',
        'center': (48.2082, 16.3738),
        'bounds_km': 20,
        'utm_zone': 33  # UTM Zone 33N
    },
    'budapest': {
        'name': 'Budapest',
        'center': (47.4979, 19.0402),
        'bounds_km': 20,
        'utm_zone': 34  # UTM Zone 34N
    },
    'amsterdam': {
        'name': 'Amsterdam',
        'center': (52.3676, 4.9041),
        'bounds_km': 20,
        'utm_zone': 31  # UTM Zone 31N
    },
    'stockholm': {
        'name': 'Stockholm',
        'center': (59.3293, 18.0686),
        'bounds_km': 20,
        'utm_zone': 34  # UTM Zone 34N
    },
    'helsinki': {
        'name': 'Helsinki',
        'center': (60.1699, 24.9384),
        'bounds_km': 20,
        'utm_zone': 35  # UTM Zone 35N
    },
    'petersburg': {
        'name': 'St. Petersburg',
        'center': (59.9311, 30.3609),
        'bounds_km': 20,
        'utm_zone': 36  # UTM Zone 36N
    },
    'milan': {
        'name': 'Milan',
        'center': (45.4642, 9.1900),
        'bounds_km': 20,
        'utm_zone': 32  # UTM Zone 32N
    },
    'munich': {
        'name': 'Munich',
        'center': (48.1351, 11.5820),
        'bounds_km': 20,
        'utm_zone': 32  # UTM Zone 32N
    },
    'copenhagen': {
        'name': 'Copenhagen',
        'center': (55.6761, 12.5683),
        'bounds_km': 20,
        'utm_zone': 33  # UTM Zone 33N (corrected)
    },
    'prague': {
        'name': 'Prague',
        'center': (50.0755, 14.4378),
        'bounds_km': 20,
        'utm_zone': 33  # UTM Zone 33N
    },
    'bratislava': {
        'name': 'Bratislava',
        'center': (48.1482, 17.1067),
        'bounds_km': 20,
        'utm_zone': 33  # UTM Zone 33N
    },
    'madrid': {
        'name': 'Madrid',
        'center': (40.4168, -3.7038),
        'bounds_km': 20,
        'utm_zone': 30  # UTM Zone 30N
    },
    'barcelona': {
        'name': 'Barcelona',
        'center': (41.3851, 2.1734),
        'bounds_km': 20,
        'utm_zone': 31  # UTM Zone 31N
    },
    'lisbon': {
        'name': 'Lisbon',
        'center': (38.7223, -9.1393),
        'bounds_km': 20,
        'utm_zone': 29  # UTM Zone 29N
    },
    'london': {
        'name': 'London',
        'center': (51.5074, -0.1278),
        'bounds_km': 20,
        'utm_zone': 30  # UTM Zone 30N
    },
    'lyon': {
        'name': 'Lyon',
        'center': (45.7640, 4.8357),
        'bounds_km': 20,
        'utm_zone': 31  # UTM Zone 31N
    },
    'minsk': {
        'name': 'Minsk',
        'center': (53.9045, 27.5590),
        'bounds_km': 20,
        'utm_zone': 35  # UTM Zone 35N
    },
    'kyiv': {
        'name': 'Kyiv',
        'center': (50.4501, 30.5234),
        'bounds_km': 20,
        'utm_zone': 36  # UTM Zone 36N
    },
    'san_francisco': {
        'name': 'San Francisco',
        'center': (37.7749, -122.4194),
        'bounds_km': 20,
        'utm_zone': 10  # UTM Zone 10N for California
    },
    'new_york': {
        'name': 'New York',
        'center': (40.7128, -74.0060),
        'bounds_km': 20,
        'utm_zone': 18  # UTM Zone 18N for New York
    },
    'los_angeles': {
        'name': 'Los Angeles',
        'center': (34.0522, -118.2437),
        'bounds_km': 20,
        'utm_zone': 11  # UTM Zone 11N for California
    },
    'chicago': {
        'name': 'Chicago',
        'center': (41.8781, -87.6298),
        'bounds_km': 20,
        'utm_zone': 16  # UTM Zone 16N for Illinois
    },
    'boston': {
        'name': 'Boston',
        'center': (42.3601, -71.0589),
        'bounds_km': 20,
        'utm_zone': 19  # UTM Zone 19N for Massachusetts
    },
    'paris': {
        'name': 'Paris',
        'center': (48.8566, 2.3522),
        'bounds_km': 20,
        'utm_zone': 31  # UTM Zone 31N
    },
    'kuala_lumpur': {
        'name': 'Kuala Lumpur',
        'center': (3.139, 101.6869),
        'bounds_km': 20,
        'utm_zone': 47  # UTM Zone 47N for Malaysia
    },
    'singapore': {
        'name': 'Singapore',
        'center': (1.3521, 103.8198),
        'bounds_km': 20,
        'utm_zone': 48  # UTM Zone 48N for Singapore
    },
    'tokyo': {
        'name': 'Tokyo',
        'center': (35.6762, 139.6503),
        'bounds_km': 20,
        'utm_zone': 54  # UTM Zone 54N for Japan
    },
    'seoul': {
        'name': 'Seoul',
        'center': (37.5665, 126.9780),
        'bounds_km': 20,
        'utm_zone': 52  # UTM Zone 52N for South Korea
    },
    'sydney': {
        'name': 'Sydney',
        'center': (-33.8688, 151.2093),
        'bounds_km': 20,
        'utm_zone': 56  # UTM Zone 56S for Australia
    },
    'melbourne': {
        'name': 'Melbourne',
        'center': (-37.8136, 144.9631),
        'bounds_km': 20,
        'utm_zone': 55  # UTM Zone 55S for Australia
    },
    'shanghai': {
        'name': 'Shanghai',
        'center': (31.2304, 121.4737),
        'bounds_km': 20,
        'utm_zone': 51  # UTM Zone 51N for China
    },
    'beijing': {
        'name': 'Beijing',
        'center': (39.9042, 116.4074),
        'bounds_km': 20,
        'utm_zone': 50  # UTM Zone 50N for China
    },
    'chongqing': {
        'name': 'Chongqing',
        'center': (29.5630, 106.5516),
        'bounds_km': 20,
        'utm_zone': 49  # UTM Zone 49N for China
    },
    'guangzhou': {
        'name': 'Guangzhou',
        'center': (23.1291, 113.2644),
        'bounds_km': 20,
        'utm_zone': 49  # UTM Zone 49N for China
    },
    'bucharest': {
        'name': 'Bucharest',
        'center': (44.4268, 26.1025),
        'bounds_km': 20,
        'utm_zone': 35  # UTM Zone 35N for Romania
    },
    'zurich': {
        'name': 'Zurich',
        'center': (47.3769, 8.5417),
        'bounds_km': 20,
        'utm_zone': 32  # UTM Zone 32N for Switzerland
    },
    'rotterdam': {
        'name': 'Rotterdam',
        'center': (51.9225, 4.4792),
        'bounds_km': 20,
        'utm_zone': 31  # UTM Zone 31N for Netherlands
    },
    'melbourne': {
        'name': 'Melbourne',
        'center': (-37.8136, 144.9631),
        'bounds_km': 20,
        'utm_zone': 55  # UTM Zone 55S for Australia
    },
}