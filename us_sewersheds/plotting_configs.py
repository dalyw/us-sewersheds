import json
import os
import pandas as pd
import folium

# Default color for missing data
DEFAULT_NODE_COLOR = "#FFFFC5"  # Light yellow


# Load facility type groups from JSON file
def load_facility_type_groups():
    """Load facility type groups from JSON file"""
    json_path = os.path.join(
        os.path.dirname(__file__), "facility_type_groups.json"
    )
    with open(json_path, "r") as f:
        return json.load(f)


FACILITY_TYPE_GROUPS = load_facility_type_groups()


def get_facility_type_order():
    """Return dictionary mapping facility types to their processing order"""
    facility_type_order = {}
    for group_data in FACILITY_TYPE_GROUPS.values():
        processing_order = group_data.get("processing_order", 999)
        for facility_type in group_data["facility_types"]:
            facility_type_order[facility_type] = processing_order
    return facility_type_order


def get_node_color(facility_type, facility_name=None):
    """
    Determine node color based on facility type and name.
    """
    # Check if facility type is in any of the groups
    for group_name, group_data in FACILITY_TYPE_GROUPS.items():
        if facility_type in group_data["facility_types"]:
            return group_data["color"]

    # Check if facility name contains keywords for special cases
    if facility_name:
        facility_name_lower = facility_name.lower()

        # Special cases based on facility name
        if any(
            keyword in facility_name_lower
            for keyword in ["outfall", "discharge", "ocean"]
        ):
            return FACILITY_TYPE_GROUPS["Outfall"][
                "color"
            ]  # Green for outfalls
        elif any(
            keyword in facility_name_lower for keyword in ["reuse", "recycling"]
        ):
            return FACILITY_TYPE_GROUPS["Water Reuse & Recovery"][
                "color"
            ]  # Purple for reuse facilities

    # Default color for unmapped types
    return DEFAULT_NODE_COLOR


def create_facility_info(facility, include_html=False):
    """Create standardized facility information for display."""
    facility_name = facility.get("FACILITY_NAME", "Unknown")
    facility_type = facility.get("FACILITY_TYPE", "Other")
    cwns_id = facility.get("CWNS_ID", "N/A")
    population = facility.get("TOTAL_RES_POPULATION_2022", "N/A")
    design_flow = facility.get("CURRENT_DESIGN_FLOW", "N/A")
    permit_number = facility.get("PERMIT_NUMBER", "N/A")
    dummy_id = facility.get("DUMMY_ID", "N/A")

    # Format population and design flow
    if pd.notna(population) and population != "N/A":
        population = (
            f"{int(population)}"
            if isinstance(population, (int, float))
            else population
        )
    if pd.notna(design_flow) and design_flow != "N/A":
        design_flow = (
            f"{int(design_flow)}"
            if isinstance(design_flow, (int, float))
            else design_flow
        )

    # Clean up permit number (remove brackets if present)
    if pd.notna(permit_number) and permit_number != "N/A":
        permit_str = str(permit_number)
        if permit_str.startswith("[") and permit_str.endswith("]"):
            permit_number = permit_str[1:-1]
        else:
            permit_number = permit_str

    if include_html:
        return f"""
        <b>{facility_name}</b><br>
        Type: {facility_type}<br>
        CWNS ID: {cwns_id}<br>
        Population Served: {population}<br>
        Design Flow: {design_flow} MGD<br>
        DUMMY ID: {dummy_id}
        """
    else:
        return {
            "name": facility_name,
            "type": facility_type,
            "cwns_id": cwns_id,
            "population": population,
            "design_flow": design_flow,
            "permit_number": permit_number,
            "dummy_id": dummy_id,
        }


def create_network_map(facilities_df, sewershed_map, sewershed_id):
    """Create interactive Folium map showing network nodes and connections
    at their actual geographic coordinates."""

    # Get nodes and connections for the sewershed
    nodes = sewershed_map[sewershed_id]["nodes"]
    connections = sewershed_map[sewershed_id]["connections"]

    # Filter facilities to only include those in this sewershed
    sewershed_facilities = facilities_df[facilities_df["DUMMY_ID"].isin(nodes)]

    if sewershed_facilities.empty:
        # Fallback to basic map if no facilities found
        center_lat = facilities_df["LATITUDE"].mean()
        center_lon = facilities_df["LONGITUDE"].mean()
        return folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles="OpenStreetMap",
        )

    # Calculate center based on sewershed facilities
    center_lat = sewershed_facilities["LATITUDE"].mean()
    center_lon = sewershed_facilities["LONGITUDE"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="OpenStreetMap",
    )

    # Create a mapping from DUMMY_ID to facility data
    facility_lookup = {}
    for _, facility in sewershed_facilities.iterrows():
        facility_lookup[facility["DUMMY_ID"]] = facility

    # Add connections first (so they appear behind nodes)
    for connection in connections:
        source_id, target_id, flow_percentage = connection

        # Get coordinates for source and target nodes
        source_facility = facility_lookup.get(source_id)
        target_facility = facility_lookup.get(target_id)

        if source_facility is not None and target_facility is not None:
            if (
                pd.notna(source_facility["LATITUDE"])
                and pd.notna(source_facility["LONGITUDE"])
                and pd.notna(target_facility["LATITUDE"])
                and pd.notna(target_facility["LONGITUDE"])
            ):

                # Calculate line thickness based on design flow of
                # downstream (target) node
                design_flow = target_facility.get("CURRENT_DESIGN_FLOW", 0)
                if pd.notna(design_flow) and design_flow > 0:
                    # Scale line thickness: min 1, max 8, based on design flow
                    line_weight = max(1, min(8, int(design_flow / 5)))
                else:
                    line_weight = 1

                # Draw connection line
                folium.PolyLine(
                    locations=[
                        [
                            source_facility["LATITUDE"],
                            source_facility["LONGITUDE"],
                        ],
                        [
                            target_facility["LATITUDE"],
                            target_facility["LONGITUDE"],
                        ],
                    ],
                    color="#666666",
                    weight=line_weight,
                    opacity=0.7,
                    popup=(
                        f"Flow: {flow_percentage}"
                        f"Design Flow: {design_flow} MGD"
                    ),
                    tooltip=(
                        f"{source_id} → {target_id}: {flow_percentage}% "
                        f"({design_flow} MGD)"
                    ),
                ).add_to(m)

    # Add facility markers (nodes)
    used_colors = set()
    for _, facility in sewershed_facilities.iterrows():
        if pd.notna(facility["LATITUDE"]) and pd.notna(facility["LONGITUDE"]):
            facility_type = facility.get("FACILITY_TYPE", "Other")
            color = get_node_color(facility_type)
            used_colors.add(color)

            # Use consistent marker size
            radius = 8

            popup_text = create_facility_info(facility, include_html=True)

            folium.CircleMarker(
                location=[facility["LATITUDE"], facility["LONGITUDE"]],
                radius=radius,
                popup=folium.Popup(popup_text, max_width=300),
                color="black",
                weight=2,
                fillColor=color,
                fillOpacity=0.8,
            ).add_to(m)

    # Add legend
    legend_html = create_network_legend(used_colors)
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def create_legend_items(used_colors):
    """Create legend items for both network map and Cytoscape graph."""
    legend_items = []
    added_groups = set()

    for color in used_colors:
        # Find the group that uses this color
        for group_name, group_data in FACILITY_TYPE_GROUPS.items():
            if group_data["color"] == color and group_name not in added_groups:
                added_groups.add(group_name)
                legend_items.append(
                    f"""
                <div style="display: flex; align-items: center; margin: 5px 0;">
                    <div style="width: 20px; height: 20px;
                                background-color: {color};
                                border: 1px solid #000; margin-right: 8px;
                                border-radius: 50%;"></div>
                    <span style="font-size: 12px;">{group_name}</span>
                </div>
                """
                )
                break

    return legend_items


def create_network_legend(used_colors):
    """Create HTML legend for network map."""
    legend_items = create_legend_items(used_colors)

    legend_html = f"""
     <div style="position: fixed; top: 10px; right: 10px; width: 200px;
                 background: white; padding: 10px; border: 1px solid #ccc;
                 border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                 z-index: 1000; font-family: Arial, sans-serif;">
         <h4 style="margin: 0 0 10px 0; font-size: 14px; text-align: center;">
             Network Legend</h4>
        {''.join(legend_items)}
        <div style="margin-top: 10px; padding-top: 10px;
                    border-top: 1px solid #ddd;">
            <div style="font-size: 11px; color: #666;">
                • Line thickness = Design flow<br>
                • Click nodes for details
            </div>
        </div>
    </div>
    """

    return legend_html


def create_cytoscape_legend(used_colors):
    """Create legend items for Cytoscape network graph."""
    legend_items = []
    added_groups = set()

    for color in used_colors:
        for group_name, group_data in FACILITY_TYPE_GROUPS.items():
            if group_data["color"] == color and group_name not in added_groups:
                added_groups.add(group_name)
                legend_items.append(
                    f"""
                    <div class="legend-item">
                        <div class="legend-color"
                             style="background-color: {color};"></div>
                        {group_name}
                    </div>
                    """
                )
                break

    return legend_items
