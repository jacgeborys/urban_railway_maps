"""
Generate a collage of transit maps in a grid layout
"""
from PIL import Image, ImageDraw, ImageFont
import os

# Configuration
ROWS = 5
COLS = 4
CELL_SIZE = 800  # pixels per map
PADDING = 20  # pixels between maps
TITLE_HEIGHT = 60  # height for city name below each map

# Selected cities in display order (top-left to bottom-right)
SELECTED_CITIES = [
    'london',
    'paris',
    'new_york',
    'tokyo',
    'warszawa',
    'berlin',
    'moscow',
    'seoul',
    'vienna',
    'amsterdam',
    'prague',
    'munich',
    'barcelona',
    'stockholm',
    'singapore',
    'melbourne',
    'istanbul',
    'toronto',
    'sao_paulo',
    'krakow',
]

# City display names (override if needed)
CITY_NAMES = {
    'warszawa': 'Warsaw',
    'krakow': 'Kraków',
    'new_york': 'New York',
    'sao_paulo': 'São Paulo',
}


def get_city_display_name(city_key):
    """Get the display name for a city."""
    if city_key in CITY_NAMES:
        return CITY_NAMES[city_key]
    # Convert key to title case
    return city_key.replace('_', ' ').title()


def create_collage(output_filename='transit_maps_collage.png'):
    """Create a collage of transit maps."""

    # Calculate final image dimensions
    img_width = COLS * CELL_SIZE + (COLS - 1) * PADDING
    img_height = ROWS * (CELL_SIZE + TITLE_HEIGHT) + (ROWS - 1) * PADDING

    # Create blank white canvas
    collage = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(collage)

    # Try to load Inter font for labels
    try:
        font_path = r"D:\QGIS\functional_map\script\font\Inter_28pt-Regular.ttf"
        font = ImageFont.truetype(font_path, 32)
    except:
        print("Warning: Could not load Inter font, using default")
        font = ImageFont.load_default()

    # Process each city
    for idx, city_key in enumerate(SELECTED_CITIES):
        if idx >= ROWS * COLS:
            break

        # Calculate grid position
        row = idx // COLS
        col = idx % COLS

        # Calculate position in collage
        x = col * (CELL_SIZE + PADDING)
        y = row * (CELL_SIZE + TITLE_HEIGHT + PADDING)

        # Load city map
        img_path = f'img/{city_key}_transit_map_clean.png'  # Add '_clean'

        if not os.path.exists(img_path):
            print(f"Warning: Map not found for {city_key} at {img_path}")
            # Draw placeholder
            draw.rectangle([x, y, x + CELL_SIZE, y + CELL_SIZE],
                           fill='#f0f0f0', outline='#cccccc')
            draw.text((x + CELL_SIZE // 2, y + CELL_SIZE // 2),
                      'Map not found', fill='#999999', anchor='mm', font=font)
        else:
            try:
                city_img = Image.open(img_path)

                # Resize to fit cell while maintaining aspect ratio
                city_img.thumbnail((CELL_SIZE, CELL_SIZE), Image.Resampling.LANCZOS)

                # Center the image in the cell
                img_x = x + (CELL_SIZE - city_img.width) // 2
                img_y = y + (CELL_SIZE - city_img.height) // 2

                collage.paste(city_img, (img_x, img_y))
                print(f"Added {city_key} at position ({row}, {col})")

            except Exception as e:
                print(f"Error loading {city_key}: {e}")

        # Add city name below the map
        city_name = get_city_display_name(city_key)
        text_y = y + CELL_SIZE + 10

        # Get text bounding box for centering
        bbox = draw.textbbox((0, 0), city_name, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x + (CELL_SIZE - text_width) // 2

        draw.text((text_x, text_y), city_name, fill='#333333', font=font)

    # Save the collage
    collage.save(output_filename, dpi=(300, 300), quality=95)
    print(f"\nCollage saved as {output_filename}")
    print(f"Dimensions: {img_width} × {img_height} pixels")
    print(f"Grid: {ROWS} rows × {COLS} columns")

    return collage


def create_compact_collage(output_filename='transit_maps_collage_3x4.png'):
    """Create a more compact 3×4 collage with 12 cities."""

    global ROWS, COLS, SELECTED_CITIES

    # Temporarily override settings
    original_rows, original_cols = ROWS, COLS
    original_cities = SELECTED_CITIES.copy()

    ROWS = 4
    COLS = 3
    SELECTED_CITIES = [
        'london', 'paris', 'tokyo',
        'new_york', 'warszawa', 'berlin',
        'moscow', 'seoul', 'barcelona',
        'singapore', 'amsterdam', 'vienna',
    ]

    result = create_collage(output_filename)

    # Restore original settings
    ROWS, COLS = original_rows, original_cols
    SELECTED_CITIES = original_cities

    return result


if __name__ == "__main__":
    import sys

    print("Transit Map Collage Generator")
    print("=" * 50)

    if len(sys.argv) > 1 and sys.argv[1] == 'compact':
        print("\nGenerating compact 3×4 collage (12 cities)...")
        create_compact_collage()
    else:
        print(f"\nGenerating full 4×5 collage (20 cities)...")
        print(f"Selected cities: {', '.join(SELECTED_CITIES)}\n")
        create_collage()

        print("\nTip: Run 'python generate_collage.py compact' for a 3×4 grid")