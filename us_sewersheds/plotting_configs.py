# Default color for missing data
DEFAULT_NODE_COLOR = '#FFFFC5'  # Light yellow

# Color mapping dictionary for different facility types
FACILITY_COLOR_MAP = {
    'Reuse': '#9370DB',                    # Purple
    'Ocean Discharge': '#90EE90',          # Green
    'Treatment Plant': '#ADD8E6',          # Blue
    'Collection: Separate Sewers': '#C4A484',      # Brown
    'Collection: Pump Stations': '#C4A484',        # Brown
    'Collection: Combined Sewers': '#C4A484',      # Brown
    'Collection: Interceptor Sewers': '#C4A484',   # Brown
    'Storage Tanks': '#808080',            # Grey
    'Storage Facility': '#808080',         # Grey
    'Onsite Wastewater Treatment System': '#FFD580',  # Orange
    'Phase II MS4': '#FFD580',            # Orange
    'Phase I MS4': '#FFD580',             # Orange
    'Non-traditional MS4': '#FFD580',     # Orange
    'Sanitary Landfills': '#FFD580',      # Orange
    'Honey Bucket Lagoon': '#FFD580',     # Orange
    'Water Reuse': '#9370DB',             # Purple
    'Resource Extraction': '#9370DB',     # Purple
    'Desalination - WW': '#9370DB',       # Purple
    'Biosolids Handling Facility': '#ADD8E6',  # Blue
    'Clustered System': '#ADD8E6',        # Blue
    'Other': '#FFFFC5'                    # Light yellow
}

def get_node_color(facility_type, facility_name):
    """
    Determine node color based on facility type and name.
    """

    # # Partial matches in facility name
    for key, color in FACILITY_COLOR_MAP.items():
        if key in facility_name or key in facility_type:
            return color
    
    # Default color for unknown types
    return DEFAULT_NODE_COLOR
