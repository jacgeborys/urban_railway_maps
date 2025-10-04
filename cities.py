# Configuration for different cities
# NOTE: 'name' is now only used for display purposes, not for OSM queries
# OSM queries use coordinates with bounding box instead

CITIES = {
    # 'warszawa': {
    #         'name': 'Warsaw',
    #         'center': (52.2297, 21.0122),
    #         'bounds_km': 25,
    #         'utm_zone': 34  # UTM Zone 34N for Poland
    #     },
    #     'gdansk': {
    #         'name': 'Gdańsk',
    #         'center': (54.4020, 18.5966),
    #         'bounds_km': 25,
    #         'utm_zone': 33  # UTM Zone 33N
    #     },
    #     'krakow': {
    #         'name': 'Kraków',
    #         'center': (50.0647, 19.9450),
    #         'bounds_km': 25,
    #         'utm_zone': 34  # UTM Zone 34N
    #     },
    #     'berlin': {
    #         'name': 'Berlin',
    #         'center': (52.5200, 13.4050),
    #         'bounds_km': 25,
    #         'utm_zone': 33  # UTM Zone 33N
    #     },
    #     'poznan': {
    #         'name': 'Poznań',
    #         'center': (52.4064, 16.9252),
    #         'bounds_km': 25,
    #         'utm_zone': 33  # UTM Zone 33N
    #     },
    #     'wroclaw': {
    #         'name': 'Wrocław',
    #         'center': (51.1079, 17.0385),
    #         'bounds_km': 25,
    #         'utm_zone': 33  # UTM Zone 33N
    #     },
    #     'vienna': {
    #         'name': 'Vienna',
    #         'center': (48.2082, 16.3738),
    #         'bounds_km': 25,
    #         'utm_zone': 33  # UTM Zone 33N
    #     },
    #     'budapest': {
    #         'name': 'Budapest',
    #         'center': (47.4979, 19.0702),
    #         'bounds_km': 25,
    #         'utm_zone': 34  # UTM Zone 34N
    #     },
    #     'amsterdam': {
    #         'name': 'Amsterdam',
    #         'center': (52.3676, 4.9041),
    #         'bounds_km': 25,
    #         'utm_zone': 31  # UTM Zone 31N
    #     },
    #     'stockholm': {
    #         'name': 'Stockholm',
    #         'center': (59.3293, 18.0686),
    #         'bounds_km': 25,
    #         'utm_zone': 34  # UTM Zone 34N
    #     },
    #     'helsinki': {
    #         'name': 'Helsinki',
    #         'center': (60.2199, 24.9384),
    #         'bounds_km': 25,
    #         'utm_zone': 35  # UTM Zone 35N
    #     },
    #     'petersburg': {
    #         'name': 'St. Petersburg',
    #         'center': (59.9311, 30.3609),
    #         'bounds_km': 25,
    #         'utm_zone': 36  # UTM Zone 36N
    #     },
    #     'milan': {
    #         'name': 'Milan',
    #         'center': (45.4642, 9.1900),
    #         'bounds_km': 25,
    #         'utm_zone': 32  # UTM Zone 32N
    #     },
    #     'munich': {
    #         'name': 'Munich',
    #         'center': (48.1351, 11.5820),
    #         'bounds_km': 25,
    #         'utm_zone': 32  # UTM Zone 32N
    #     },
    #     'copenhagen': {
    #         'name': 'Copenhagen',
    #         'center': (55.6961, 12.5383),
    #         'bounds_km': 25,
    #         'utm_zone': 33  # UTM Zone 33N (corrected)
    #     },
    #     'prague': {
    #         'name': 'Prague',
    #         'center': (50.0755, 14.4378),
    #         'bounds_km': 25,
    #         'utm_zone': 33  # UTM Zone 33N
    #     },
    #     'bratislava': {
    #         'name': 'Bratislava',
    #         'center': (48.1482, 17.1067),
    #         'bounds_km': 25,
    #         'utm_zone': 33  # UTM Zone 33N
    #     },
    #     'madrid': {
    #         'name': 'Madrid',
    #         'center': (40.4168, -3.7038),
    #         'bounds_km': 25,
    #         'utm_zone': 30  # UTM Zone 30N
    #     },
    #     'barcelona': {
    #         'name': 'Barcelona',
    #         'center': (41.3851, 2.1534),
    #         'bounds_km': 25,
    #         'utm_zone': 31  # UTM Zone 31N
    #     },
    #     'lisbon': {
    #         'name': 'Lisbon',
    #         'center': (38.7223, -9.1393),
    #         'bounds_km': 25,
    #         'utm_zone': 29  # UTM Zone 29N
    #     },
    #     'london': {
    #         'name': 'London',
    #         'center': (51.5074, -0.1278),
    #         'bounds_km': 25,
    #         'utm_zone': 30  # UTM Zone 30N
    #     },
    #     'lyon': {
    #         'name': 'Lyon',
    #         'center': (45.7640, 4.8357),
    #         'bounds_km': 25,
    #         'utm_zone': 31  # UTM Zone 31N
    #     },
    #     'minsk': {
    #         'name': 'Minsk',
    #         'center': (53.9045, 27.5590),
    #         'bounds_km': 25,
    #         'utm_zone': 35  # UTM Zone 35N
    #     },
    #     'kyiv': {
    #         'name': 'Kyiv',
    #         'center': (50.4501, 30.5234),
    #         'bounds_km': 25,
    #         'utm_zone': 36  # UTM Zone 36N
    #     },
    #     'san_francisco': {
    #         'name': 'San Francisco',
    #         'center': (37.7749, -122.3754),
    #         'bounds_km': 25,
    #         'utm_zone': 10  # UTM Zone 10N for California
    #     },
    #     'new_york': {
    #         'name': 'New York',
    #         'center': (40.7128, -74.0060),
    #         'bounds_km': 25,
    #         'utm_zone': 18  # UTM Zone 18N for New York
    #     },
    #     'los_angeles': {
    #         'name': 'Los Angeles',
    #         'center': (34.0522, -118.2437),
    #         'bounds_km': 25,
    #         'utm_zone': 11  # UTM Zone 11N for California
    #     },
    #     'chicago': {
    #         'name': 'Chicago',
    #         'center': (41.8781, -87.6998),
    #         'bounds_km': 25,
    #         'utm_zone': 16  # UTM Zone 16N for Illinois
    #     },
    #     'boston': {
    #         'name': 'Boston',
    #         'center': (42.3601, -71.0589),
    #         'bounds_km': 25,
    #         'utm_zone': 19  # UTM Zone 19N for Massachusetts
    #     },
        'paris': {
            'name': 'Paris',
            'center': (48.8566, 2.3522),
            'bounds_km': 25,
            'utm_zone': 31  # UTM Zone 31N
        },
        'kuala_lumpur': {
            'name': 'Kuala Lumpur',
            'center': (3.139, 101.6869),
            'bounds_km': 25,
            'utm_zone': 47  # UTM Zone 47N for Malaysia
        },
        'singapore': {
            'name': 'Singapore',
            'center': (1.3521, 103.8198),
            'bounds_km': 30,
            'utm_zone': 48  # UTM Zone 48N for Singapore
        },
        'tokyo': {
            'name': 'Tokyo',
            'center': (35.6762, 139.7603),
            'bounds_km': 25,
            'utm_zone': 54  # UTM Zone 54N for Japan
        },
        'seoul': {
            'name': 'Seoul',
            'center': (37.5665, 126.9780),
            'bounds_km': 25,
            'utm_zone': 52  # UTM Zone 52N for South Korea
        },
        'sydney': {
            'name': 'Sydney',
            'center': (-33.8688, 151.0913),
            'bounds_km': 35,
            'utm_zone': 56,
            'hemisphere': 'S'  # Specify Southern Hemisphere
        },
        'melbourne': {
            'name': 'Melbourne',
            'center': (-37.8136, 144.9631),
            'bounds_km': 25,
            'utm_zone': 55,
            'hemisphere': 'S'  # Specify Southern Hemisphere
        },
        'shanghai': {
            'name': 'Shanghai',
            'center': (31.2304, 121.4737),
            'bounds_km': 25,
            'utm_zone': 51  # UTM Zone 51N for China
        },
        'beijing': {
            'name': 'Beijing',
            'center': (39.9042, 116.4074),
            'bounds_km': 25,
            'utm_zone': 50  # UTM Zone 50N for China
        },
        'chongqing': {
            'name': 'Chongqing',
            'center': (29.5630, 106.5516),
            'bounds_km': 25,
            'utm_zone': 49  # UTM Zone 49N for China
        },
        'guangzhou': {
            'name': 'Guangzhou',
            'center': (23.1291, 113.2644),
            'bounds_km': 25,
            'utm_zone': 49  # UTM Zone 49N for China
        },
        'bucharest': {
            'name': 'Bucharest',
            'center': (44.4268, 26.1025),
            'bounds_km': 25,
            'utm_zone': 35  # UTM Zone 35N for Romania
        },
        'zurich': {
            'name': 'Zurich',
            'center': (47.3769, 8.5417),
            'bounds_km': 25,
            'utm_zone': 32  # UTM Zone 32N for Switzerland
        },
        'rotterdam': {
            'name': 'Rotterdam',
            'center': (51.9225, 4.4792),
            'bounds_km': 25,
            'utm_zone': 31  # UTM Zone 31N for Netherlands
        },
        'buenosaires': {
            'name': 'Buenos Aires',
            'center': (-34.6037, -58.4716),
            'bounds_km': 25,
            'utm_zone': 21,  # UTM Zone 21S for Argentina
            'hemisphere': 'S'  # Specify Southern Hemisphere
        },
        'sao_paulo': {
            'name': 'São Paulo',
            'center': (-23.5505, -46.6333),
            'bounds_km': 25,
            'utm_zone': 23,  # UTM Zone 23S for Brazil
            'hemisphere': 'S'  # Specify Southern Hemisphere
        },
        'dublin': {
            'name': 'Dublin',
            'center': (53.3498, -6.2603),
            'bounds_km': 25,
            'utm_zone': 29  # UTM Zone 29N for Ireland
        },
        'oslo': {
            'name': 'Oslo',
            'center': (59.9139, 10.7522),
            'bounds_km': 25,
            'utm_zone': 33  # UTM Zone 33N for Norway
        },
        'zagreb': {
            'name': 'Zagreb',
            'center': (45.8150, 15.9819),
            'bounds_km': 25,
            'utm_zone': 33  # UTM Zone 33N for Croatia
        },
        'hamburg': {
            'name': 'Hamburg',
            'center': (53.5511, 9.9937),
            'bounds_km': 25,
            'utm_zone': 32  # UTM Zone 32N for Germany
        },
        'moscow': {
            'name': 'Moscow',
            'center': (55.7558, 37.6173),
            'bounds_km': 25,
            'utm_zone': 37  # UTM Zone 37N for Russia
        },
        'valencia': {
            'name': 'Valencia',
            'center': (39.4699, -0.3763),
            'bounds_km': 25,
            'utm_zone': 30  # UTM Zone 30N for Spain
        },
        'naples': {
            'name': 'Naples',
            'center': (40.8518, 14.2681),
            'bounds_km': 25,
            'utm_zone': 33  # UTM Zone 33N for Italy
        },
        'roma': {
            'name': 'Rome',
            'center': (41.9028, 12.4964),
            'bounds_km': 25,
            'utm_zone': 33  # UTM Zone 33N for Italy
        },
    'toronto': {
        'name': 'Toronto',
        'center': (43.711070, -79.377015),
        'bounds_km': 25,
        'utm_zone': 17  # UTM Zone 17N for Ontario, Canada
    },
    'vancouver': {
        'name': 'Vancouver',
        'center': (49.2827, -123.0257),
        'bounds_km': 25,
        'utm_zone': 10  # UTM Zone 10N for British Columbia, Canada
    },
    'istanbul': {
        'name': 'Istanbul',
        'center': (41.0382, 28.9784),
        'bounds_km': 25,
        'utm_zone': 35  # UTM Zone 35N for Turkey
    },
    'montreal': {
        'name': 'Montreal',
        'center': (45.5017, -73.5673),
        'bounds_km': 25,
        'utm_zone': 18  # UTM Zone 18N for Quebec, Canada
    },
    'newdelhi': {
        'name': 'New Delhi',
        'center': (28.6139, 77.2090),
        'bounds_km': 25,
        'utm_zone': 43  # UTM Zone 43N for India
    },
    'bogota': {
        'name': 'Bogotá',
        'center': (4.7110, -74.0721),
        'bounds_km': 25,
        'utm_zone': 18,  # UTM Zone 18N for Colombia
    },
    'casablanca': {
        'name': 'Casablanca',
        'center': (33.5731, -7.5898),
        'bounds_km': 25,
        'utm_zone': 29  # UTM Zone 29N for Morocco
    },
    'cairo': {
        'name': 'Cairo',
        'center': (30.0444, 31.2357),
        'bounds_km': 25,
        'utm_zone': 36  # UTM Zone 36N for Egypt
    },
    'jakarta': {
        'name': 'Jakarta',
        'center': (-6.2088, 106.8456),
        'bounds_km': 25,
        'utm_zone': 48,  # UTM Zone 48S for Indonesia
        'hemisphere': 'S'  # Specify Southern Hemisphere
    },
    'riyadh': {
        'name': 'Riyadh',
        'center': (24.7136, 46.7153),
        'bounds_km': 40,
        'utm_zone': 38  # UTM Zone 38N for Saudi Arabia
    },
    'mexicocity': {
        'name': 'Mexico City',
        'center': (19.4326, -99.1332),
        'bounds_km': 25,
        'utm_zone': 14  # UTM Zone 14N for Mexico
    },
    'santiago': {
        'name': 'Santiago',
        'center': (-33.4489, -70.6693),
        'bounds_km': 25,
        'utm_zone': 19,  # UTM Zone 19S for Chile
        'hemisphere': 'S'  # Specify Southern Hemisphere
    },
    'silesia': {
        'name': 'Silesia',
        'center': (50.2649, 19.0238),  # Katowice center
        'bounds_km': 40,  # Larger area to cover multiple cities
        'utm_zone': 34  # UTM Zone 34N for Poland
    },
    'lodz': {
        'name': 'Łódź',
        'center': (51.7592, 19.4560),
        'bounds_km': 25,
        'utm_zone': 34  # UTM Zone 34N for Poland
    },
    'szczecin': {
        'name': 'Szczecin',
        'center': (53.4285, 14.5528),
        'bounds_km': 25,
        'utm_zone': 33  # UTM Zone 33N for Poland
    },
    'washington': {
        'name': 'Washington, D.C.',
        'center': (38.9072, -77.0369),
        'bounds_km': 25,
        'utm_zone': 18  # UTM Zone 18N for Washington, D.C.
    },
    'philadelphia': {
        'name': 'Philadelphia',
        'center': (39.9526, -75.1652),
        'bounds_km': 25,
        'utm_zone': 18  # UTM Zone 18N for Pennsylvania
    },
    'glasgow': {
        'name': 'Glasgow',
        'center': (55.8642, -4.2518),
        'bounds_km': 25,
        'utm_zone': 30  # UTM Zone 30N for Scotland
    },
    'torun': {
        'name': 'Toruń',
        'center': (53.0138, 18.5984),
        'bounds_km': 15,
        'utm_zone': 34  # UTM Zone 34N for Poland
    },
    'dubai': {
        'name': 'Dubai',
        'center': (25.1448, 55.2708),
        'bounds_km': 40,
        'utm_zone': 40  # UTM Zone 40N for UAE
    },
    'brussels': {
        'name': 'Brussels',
        'center': (50.8503, 4.3517),
        'bounds_km': 25,
        'utm_zone': 31  # UTM Zone 31N for Belgium
    },
    'athens': {
        'name': 'Athens',
        'center': (37.9838, 23.7275),
        'bounds_km': 25,
        'utm_zone': 34  # UTM Zone 34N for Greece
    },
    'stuttgart': {
        'name': 'Stuttgart',
        'center': (48.7758, 9.1829),
        'bounds_km': 25,
        'utm_zone': 32  # UTM Zone 32N for Germany
    },

}