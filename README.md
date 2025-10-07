Urban Railway Maps Generator

Automated pipeline for generating standardized transit maps from OpenStreetMap data for cities worldwide.

QUICK START

Install dependencies:
pip install matplotlib numpy shapely pyproj requests Pillow

Generate a city map:
python main.py warsaw

Generate with clean version for collages:
python main.py warsaw --clean

Generate all cities:
python main.py all


HOW IT WORKS

The pipeline:
1. Fetches transit data from OpenStreetMap's Overpass API
2. Transforms coordinates to the correct UTM projection for each city
3. Classifies routes (metro, tram, light rail, heavy rail)
4. Filters stops by proximity to lines
5. Detects and visualizes transfer stations
6. Generates labeled maps and optional clean versions


PROJECT FILES

main.py - Main script
cities.py - City configurations (coordinates, UTM zones)
osm_config.py - OSM query building and route classification
geometry_config.py - Transfer distances and visual styling
route_exceptions.py - Handling for misclassified routes
generate_collage.py - Creates grid collages from multiple cities
test_debug_stops.py - Debug tool for analyzing stop detection


ADDING A NEW CITY

1. Open cities.py
2. Add your city with its coordinates and UTM zone
3. Run: python main.py your_city

Example format in cities.py:

'lisbon': {
    'name': 'Lisbon',
    'center': (38.7223, -9.1393),
    'bounds_km': 25,
    'utm_zone': 29
}

For Southern Hemisphere cities, add: 'hemisphere': 'S'


HANDLING EDGE CASES

Some OSM routes may be misclassified. To fix:

1. Open route_exceptions.py
2. Add problematic relation IDs to EXCLUDED_RELATIONS to hide them
3. Or add to RECLASSIFIED_RELATIONS to change their type

Find relation IDs by running the debug script or checking OpenStreetMap.org


CONFIGURATION

Adjust transfer detection distances in geometry_config.py:
- Controls how close stops need to be to show as a transfer
- Different values for metro-tram, tram-tram, etc.

Customize visual styling in osm_config.py:
- Line colors, widths, opacity
- Stop marker sizes and colors
- Transfer polygon appearance


CREATING COLLAGES

After generating clean versions of cities:

python generate_collage.py

Edit SELECTED_CITIES in generate_collage.py to choose which cities to include.


FONTS

The script uses Inter font. Update paths in main.py if needed:

font_regular_path = r"path/to/Inter_28pt-Regular.ttf"
font_bold_path = r"path/to/Inter_28pt-Bold.ttf"

Download from https://fonts.google.com/specimen/Inter


API NOTES

The Overpass API has rate limits. The script automatically:
- Retries failed requests with exponential backoff
- Adds delays between requests
- Waits longer for server timeout errors

Processing all cities takes 1-2 hours.


OUTPUTS

img/ folder contains:
- Labeled maps with titles, legends, scale bars
- clean/ subfolder with label-free versions (when using --clean flag)


DEBUGGING

To investigate stop detection issues:

python test_debug_stops.py

Edit the coordinates in the file to check your specific location.


CONTRIBUTING

Contributions welcome! Open an issue or pull request.


LICENSE

MIT License


ACKNOWLEDGMENTS

OpenStreetMap contributors for the data
Reddit community for edge case feedback
Overpass API maintainers
