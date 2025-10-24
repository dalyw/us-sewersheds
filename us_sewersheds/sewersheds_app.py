import streamlit as st
import pandas as pd
import json
import streamlit.components.v1 as components
import dash_cytoscape as cyto
from dash import Dash
from us_sewersheds.helpers import (
    FACILITY_TYPE_GROUPS,
    OUTPUT_COLUMNS,
)

cyto.load_extra_layouts()

app = Dash()
server = app.server

# Set page config
st.set_page_config(
    page_title="U.S. Sewersheds",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add CSS to ensure content is visible
st.markdown(
    """
    <style>
        .main {
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        .stApp {
            background-color: white;
        }
        iframe {
            width: 100%;
            height: 600px;
            border: none;
        }
    </style>
""",
    unsafe_allow_html=True,
)

facilities = pd.read_csv("processed_data/facilities_merged.csv")[OUTPUT_COLUMNS]


def add_newlines(text, max_length=20):
    """Add newlines after spaces for text longer than max_length"""
    if len(text) <= max_length:
        return text

    space_pos = text.find(" ", max_length)
    if space_pos == -1:
        return text

    return (
        text[:space_pos]
        + "\n"
        + add_newlines(text[space_pos + 1 :], max_length)
    )


def create_facility_info(facility, include_html=False):
    """Create standardized facility information for display."""
    facility_name = facility["FACILITY_NAME"]
    facility_type = facility["FACILITY_TYPE"]
    cwns_id = facility["CWNS_ID"]
    permit_number = facility["PERMIT_NUMBER"]

    # Format population and design flow
    formatted_vals = {}
    for format_key in ["TOTAL_RES_POPULATION_2022", "CURRENT_DESIGN_FLOW"]:
        val = facility[format_key]
        if pd.notna(val) and val != "N/A":
            val = f"{int(val)}" if isinstance(val, (int, float)) else val
        formatted_vals[format_key] = val

    if include_html:
        return f"""
        <b>{facility_name}</b><br>
        Type: {facility_type}<br>
        CWNS ID: {cwns_id}<br>
        Population Served: {formatted_vals["TOTAL_RES_POPULATION_2022"]}<br>
        Design Flow: {formatted_vals[ "CURRENT_DESIGN_FLOW"]} MGD<br>
        """
    else:
        return {
            "name": facility_name,
            "type": facility_type,
            "cwns_id": cwns_id,
            "population": formatted_vals["TOTAL_RES_POPULATION_2022"],
            "design_flow": formatted_vals["CURRENT_DESIGN_FLOW"],
            "permit_number": permit_number,
        }


def create_network_map(sewershed_map, sewershed_id):
    """Create interactive Folium map showing network nodes and connections
    at their actual geographic coordinates."""

    try:
        import folium  # Lazy import folium to avoid dependency issues
    except ImportError:
        raise ImportError(
            "folium is required for map. Install with: pip install folium"
        )

    # Get nodes and connections for the sewershed
    nodes = list(sewershed_map[sewershed_id]["nodes"].keys())
    connections = sewershed_map[sewershed_id]["connections"]

    # Calculate center based on sewershed facilities
    coords = [
        (
            sewershed_map[sewershed_id]["nodes"][node_id]["LATITUDE"],
            sewershed_map[sewershed_id]["nodes"][node_id]["LONGITUDE"],
        )
        for node_id in nodes
    ]
    center_lat = sum(coord[0] for coord in coords) / len(coords)
    center_lon = sum(coord[1] for coord in coords) / len(coords)
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="OpenStreetMap",
    )

    # Add connections first (so they appear behind nodes)
    for connection in connections:
        source_id, target_id, flow_percentage = connection

        # Extract base facility IDs (remove _type_X suffix)
        source_facility = source_id.split("_type_")[0]
        target_facility = target_id.split("_type_")[0]

        # Skip self-loops within the same facility
        if source_facility == target_facility:
            continue

        # Get coordinates directly from sewershed_map
        source_data = sewershed_map[sewershed_id]["nodes"][source_facility]
        target_data = sewershed_map[sewershed_id]["nodes"][target_facility]

        if (
            pd.notna(source_data.get("LATITUDE"))
            and pd.notna(source_data.get("LONGITUDE"))
            and pd.notna(target_data.get("LATITUDE"))
            and pd.notna(target_data.get("LONGITUDE"))
        ):

            source_coords = [source_data["LATITUDE"], source_data["LONGITUDE"]]
            target_coords = [target_data["LATITUDE"], target_data["LONGITUDE"]]

            # Draw connection line
            folium.PolyLine(
                locations=[source_coords, target_coords],
                color="#666666",
                weight=2,
                opacity=0.7,
                popup=(
                    f"{flow_percentage}%" if flow_percentage is not None else ""
                ),
            ).add_to(m)

    # Add facility markers (nodes)
    used_colors = set()
    for node_id in nodes:
        node_data = sewershed_map[sewershed_id]["nodes"][node_id]
        if pd.notna(node_data.get("LATITUDE")) and pd.notna(
            node_data.get("LONGITUDE")
        ):
            types = node_data["TYPES"]

            # Determine color: single type uses type color, multiple types use grey
            if len(types) == 1:
                color = list(types.values())[0]["color"]
                facility_types = [list(types.values())[0]["FACILITY_TYPE"]]
            else:
                color = "#808080"  # Grey for multiple types
                facility_types = [ft["FACILITY_TYPE"] for ft in types.values()]

            used_colors.add(color)

            # Create popup text for this facility
            facility_name = node_data["FACILITY_NAME"]

            popup_text = f"""
            <b>{facility_name}</b><br>
            CWNS ID: {node_data['CWNS_ID']}<br>
            Types: {', '.join(facility_types)}<br>
            Population Served: {node_data["TOTAL_RES_POPULATION_2022"]}<br>
            Design Flow: {node_data["CURRENT_DESIGN_FLOW"]} MGD
            """

            folium.CircleMarker(
                location=[node_data["LATITUDE"], node_data["LONGITUDE"]],
                popup=popup_text,
                color="black",
                fillColor=color,
            ).add_to(m)

    # Add legend
    legend_html = create_legend(used_colors, html=True)
    m.get_root().html.add_child(folium.Element(legend_html))

    return m._repr_html_()


def create_legend(used_colors, html=False):
    """Create legend items for Cytoscape network graph."""

    legend_items = []
    added_groups = set()
    for color in used_colors:
        if color == "#808080":  # Grey for multiple types
            legend_items.append(
                f'<div><div class="legend-color" style="background-color: {color};"></div>Multiple Types</div>'
            )
        else:
            for name, group_data in FACILITY_TYPE_GROUPS.items():
                if group_data["color"] == color and name not in added_groups:
                    added_groups.add(name)
                    legend_items.append(
                        f'<div><div class="legend-color" style="background-color: {color};"></div>{name}</div>'
                    )

    if html:
        return f'<div style="position: fixed; top: 10px; right: 10px; background: white; padding: 10px; border: 1px solid #ccc;">{"".join(legend_items)}</div>'
    else:
        return legend_items


def plot_sewershed(sewershed_id, sewershed_map, facilities):
    """
    Each entry in sewershed_map is a dictionary with keys "nodes" and 'connections'.
    Plots a directed graph of a given sewershed using Cytoscape for interactive visualization.
    Now handles nested facility types structure.

    Inputs:
    - sewershed_id: string, the ID of the sewershed to plot
    - sewershed_map: dictionary, the sewershed map
    - facilities: pandas dataframe, the facilities dataframe

    Outputs:
    - HTML component
    """
    nodes = list(sewershed_map[sewershed_id]["nodes"].keys())
    connections = sewershed_map[sewershed_id]["connections"]
    node_data = sewershed_map[sewershed_id]["nodes"]

    elements = []
    used_colors = set()

    # Find max population in network
    max_pop = 0
    for node in nodes:
        facility_mask = facilities["CWNS_ID"] == node
        if not facilities[facility_mask].empty:
            population = facilities.loc[
                facility_mask, "TOTAL_RES_POPULATION_2022"
            ].iloc[0]
            if pd.notna(population) and population > max_pop:
                max_pop = population

    for node in nodes:
        # Get node data from sewershed map
        node_data = sewershed_map[sewershed_id]["nodes"].get(node, {})

        # Always create nodes for each facility type
        for i, facility_type in node_data["TYPES"].items():
            node_id = f"{node}_type_{i}"
            name = f"{node_data.get('FACILITY_NAME', node)} ({facility_type['FACILITY_TYPE']})"

            if len(name) > 20:  # new lines after every 16 chars
                name = add_newlines(name)

            # Create data dict with all node_data keys plus specific ones
            data_dict = {
                "id": node_id,
                "label": name,
                "color": facility_type["color"],
                "shape": facility_type["shape"],
                "CWNS_ID": node,
                "FACILITY_TYPE": facility_type["FACILITY_TYPE"],
            }
            used_colors.add(data_dict["color"])

            # Add all keys from node_data
            for key, value in node_data.items():
                if key != "TYPES":  # Skip TYPES as it's handled separately
                    data_dict[key] = value

            elements.append({"data": data_dict})

    for i, conn in enumerate(connections):
        elements.append(
            {
                "data": {
                    "id": f"edge_{i}",
                    "source": str(conn[0]),
                    "target": str(conn[1]),
                    "label": f"{conn[2]}%",
                }
            }
        )

    # Build legend items based on used colors
    legend_items = create_legend(used_colors)

    cyto_html = f"""

    <html>
        <head>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
            <style>
                #cy {{ width: 100%; height: 600px; }}
                #legend {{ position: absolute; top: 10px; right: 10px; background: white; padding: 10px; border: 1px solid #ccc; }}
                .legend-color {{ display: inline-block; width: 20px; height: 20px; margin-right: 5px; }}
                #info-display {{ position: absolute; bottom: 10px; left: 10px; background: white; padding: 10px; border: 1px solid #ccc; display: none; }}
            </style>
        </head>
        <body>
            <div id="cy"></div>
            <div id="legend">
                {''.join(legend_items)}
            </div>
            <div id="info-display"></div>
            <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    var cy = cytoscape({{
                        container: document.getElementById('cy'),
                        elements: {json.dumps(elements)},
                        style: [
                            {{
                                selector: 'node',
                                style: {{
                                    'label': 'data(label)',
                                    'text-wrap': 'wrap',
                                    'background-color': 'data(color)',
                                    'shape': 'data(shape)',
                                    'color': '#000',
                                    'font-size': '12px',
                                    'text-valign': 'center',
                                    'text-halign': 'center'
                                }}
                            }},
                            {{
                                selector: 'edge',
                                style: {{
                                    'width': 2,
                                    'line-color': '#CCCCCC',
                                    'curve-style': 'bezier',
                                    'target-arrow-shape': 'triangle',
                                    'target-arrow-color': '#CCCCCC',
                                    'label': 'data(label)',
                                    'font-size': '10px',
                                    'text-rotation': 'autorotate',
                                    'text-margin-y': -10
                                }}
                            }}
                        ],
                        layout: {{
                            name: 'dagre',
                            rankDir: 'LR',
                            nodeSep: 10,
                            edgeSep: 50,
                            rankSep: 20,
                            padding: 30,
                            animate: false,
                            fit: true,
                            spacingFactor: 1.3,
                            nodeDimensionsIncludeLabels: true,
                            // Add slight waterfall effect with nodes on left higher than right
                            transform: function(node, pos) {{
                                // Calculate a vertical offset based on the horizontal position
                                // The further right, the lower the node
                                return {{
                                    x: pos.x,
                                    y: pos.y + (pos.x * 0.1) // Adjust the 0.2 multiplier to control the slope
                                }};
                            }}
                        }},
                        minZoom: 0.2,
                        maxZoom: 3
                    }});
                    var infoDisplay = document.getElementById('info-display');

                    cy.on('tap', 'node', function(evt){{
                        var node = evt.target;
                        infoDisplay.innerHTML = 'Population: ' + node.data('TOTAL_RES_POPULATION_2022') + '<br>Flow: ' + node.data('CURRENT_DESIGN_FLOW') + ' MGD';
                        infoDisplay.style.display = 'block';
                    }});

                    cy.on('tap', function(evt){{
                        if(evt.target === cy){{
                            infoDisplay.style.display = 'none';
                        }}
                    }});
                }});
            </script>
        </body>
    </html>
    """
    return cyto_html


# Load sewershed map
with open("processed_data/sewershed_map.json", "r") as f:
    sewershed_map = json.load(f)

st.title("U.S. Sewershed Network Visualization")
st.markdown("### Generate U.S. sewershed maps")

# Add view selection
view_option = st.radio(
    "Select view:", ["Network Graph", "Network Map"], horizontal=True
)

# Get states and counties
states = sorted(
    list(
        set(
            [
                name.split(" - ")[0]
                for name in sewershed_map.keys()
                if " - " in name and name.split(" - ")[0] != "Unspecified"
            ]
        )
    )
)
states.insert(0, "All States")

# Filters in a single row with buffer columns
buffer1, col1, col2, col3, buffer2 = st.columns([1, 3, 3, 3, 1])
with col1:
    selected_state = st.selectbox("Select state:", states, key="state_select")
with col2:
    counties = []
    if selected_state != "All States":
        counties = sorted(
            list(
                set(
                    [
                        name.split(" - ")[1].split(" Sewershed")[0]
                        for name in sewershed_map.keys()
                        if " - " in name
                        and name.split(" - ")[0] == selected_state
                    ]
                )
            )
        )
        counties.insert(0, "All Counties")
        selected_county = st.selectbox(
            "Select county:", counties, key="county_select"
        )
    else:
        selected_county = None
with col3:
    keyword = st.text_input(
        "Filter by facility name or permit number:", key="keyword_input"
    )

# Pre-compute keyword matches for all facilities for fast lookup
facilities_with_keyword = set()
if keyword:
    name_match_mask = facilities["FACILITY_NAME"].str.contains(
        keyword, case=False, na=False
    )
    permit_match_mask = facilities["PERMIT_NUMBER"].str.contains(
        keyword, case=False, na=False
    )
    keyword_match_mask = name_match_mask | permit_match_mask
    facilities_with_keyword = set(
        facilities.loc[keyword_match_mask, "CWNS_ID"].astype(str)
    )

# Filter results
results_list = []
for sewershed_id in sewershed_map.keys():
    if " - " not in sewershed_id:
        continue

    state = sewershed_id.split(" - ")[0]
    county = sewershed_id.split(" - ")[1].split(" Sewershed")[0]

    state_match = selected_state == "All States" or state == selected_state
    county_match = (
        not selected_county
        or selected_county == "All Counties"
        or county == selected_county
    )

    keyword_match = True
    if keyword:
        # Set intersection
        sewershed_facilities = set(sewershed_map[sewershed_id]["nodes"].keys())
        keyword_match = bool(sewershed_facilities & facilities_with_keyword)

    if state_match and county_match and keyword_match:
        results_list.append(sewershed_id)

buffer3, col4, buffer4 = st.columns([1, 9, 1])
with col4:
    dropdown = st.selectbox(
        "Select a sewershed:",
        sorted(results_list) if results_list else ["No matching sewersheds"],
    )

if dropdown != "No matching sewersheds":
    if view_option == "Network Graph":
        plot = plot_sewershed(dropdown, sewershed_map, facilities)
    elif view_option == "Network Map":
        plot = create_network_map(sewershed_map, dropdown)

    try:
        components.html(
            plot,
            height=750,
            scrolling=False,
        )
    except Exception as e:
        st.error(f"Error plotting sewershed: {e}")


st.markdown(
    """
This tool visualizes sewers, treatment facilities, outfalls, and connections as described in the 2022 Clean Watersheds Needs Survey dataset.
Data was downloaded from the "[Nationwide 2022 CWNS Dataset](https://sdwis.epa.gov/ords/sfdw_pub/r/sfdw/cwns_pub/data-download)".

**Similar methodology available in the [USEPA Sewersheds repository](https://github.com/USEPA/Sewersheds/blob/main/functions/route_h3.R).

This tool should be used for approximation and guidance only, and may not reflect the most recent or accurate depictions of any particular sewershed.
For the most up-to-date information, confirm with local or state authorities.
"""
)
