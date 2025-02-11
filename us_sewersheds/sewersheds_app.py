import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import streamlit.components.v1 as components
import dash_cytoscape as cyto
from dash import Dash, html

# Set page config
st.set_page_config(page_title="U.S. Sewersheds", layout="wide", initial_sidebar_state="expanded")

# Add CSS to ensure content is visible
st.markdown("""
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
""", unsafe_allow_html=True)

# Load data
facilities = pd.read_csv('processed_data/cwns_facilities_merged.csv')[['CWNS_ID', 'FACILITY_NAME','PERMIT_NUMBER','TOTAL_RES_POPULATION_2022','FACILITY_TYPE']]

def plot_sewershed(sewershed_id, sewershed_map, facilities):
    """
    Each entry in sewershed_map is a dictionary with keys 'nodes' and 'connections'.
    Plots a directed graph of a given sewershed using Cytoscape for interactive visualization

    Inputs:
    - sewershed_id: string, the ID of the sewershed to plot
    - sewershed_map: dictionary, the sewershed map  
    - facilities: pandas dataframe, the facilities dataframe

    Outputs:
    - HTML component
    """
    nodes = sewershed_map[sewershed_id]['nodes']
    connections = sewershed_map[sewershed_id]['connections']
    elements = []
    
    # Define color mapping for facility types, sorted by color
    facility_colors = {
        # Black - other
        'Brownfields': '#000000',
        'Estuary Management': '#000000', 
        'Marinas': '#000000',
        'Silviculture': '#000000',
        'Agriculture - Animals': '#000000',
        'Agriculture - Cropland': '#000000',
        'Ground Water - Unknown Source': '#000000',
        'Other': '#000000',

        # Blue - treatment
        'Treatment Plant': '#1f77b4',
        'Biosolids Handling Facility': '#1f77b4',
        'Clustered System': '#1f77b4',

        # Brown - collection
        'Collection: Separate Sewers': '#8B4513',
        'Collection: Pump Stations': '#8B4513',
        'Collection: Combined Sewers': '#8B4513', 
        'Collection: Interceptor Sewers': '#8B4513',

        # Grey - storage
        'Storage Tanks': '#808080',
        'Storage Facility': '#808080',

        # Orange - OWTS
        'Onsite Wastewater Treatment System': '#FFA500',
        'Phase II MS4': '#FFA500',
        'Phase I MS4': '#FFA500',
        'Non-traditional MS4': '#FFA500',
        'Sanitary Landfills': '#FFA500',
        'Honey Bucket Lagoon': '#FFA500',

        # Purple - reuse/resourec recovery
        'Water Reuse': '#9370DB',
        'Resource Extraction': '#9370DB',
        'Desalination - WW': '#9370DB',

        # Brown - stormwater
        'Unregulated Community Stormwater': '#8B4513',
        'Hydromodification': '#8B4513'
    }
    
    # Track which colors are used
    used_colors = set()
    
    for node in nodes:
        facility_mask = facilities['CWNS_ID'] == node
        if not facilities[facility_mask].empty:
            name = facilities.loc[facility_mask, 'FACILITY_NAME'].iloc[0]
            # Add newline after first space after 16 chars
            if len(name) > 16:
                space_pos = name.find(' ', 16)
                if space_pos != -1:
                    name = name[:space_pos] + '\n' + name[space_pos+1:]
            facility_type = facilities.loc[facility_mask, 'FACILITY_TYPE'].iloc[0]
            color = facility_colors.get(facility_type, facility_colors['Other'])
            used_colors.add(color)
            # Set shape to diamond if facility type contains "collection"
            shape = 'diamond' if facility_type and 'collection' in facility_type.lower() else 'ellipse'
        else:
            name = str(node)
            color = facility_colors['Other']
            used_colors.add(color)
            shape = 'ellipse'
            
        elements.append({
            'data': {
                'id': str(node),
                'label': name,
                'color': color,
                'shape': shape
            }
        })

    for conn in connections:
        elements.append({
            'data': {
                'source': str(conn[0]),
                'target': str(conn[1])
            }
        })

    # Build legend items based on used colors
    legend_items = []
    if '#1f77b4' in used_colors:
        legend_items.append("""
            <div class="legend-item">
                <div class="legend-color" style="background-color: #1f77b4;"></div>
                Centralized Treatment
            </div>
        """)
    if '#8B4513' in used_colors:
        legend_items.append("""
            <div class="legend-item">
                <div class="legend-color" style="background-color: #8B4513;"></div>
                Collection
            </div>
        """)
    if '#808080' in used_colors:
        legend_items.append("""
            <div class="legend-item">
                <div class="legend-color" style="background-color: #808080;"></div>
                Storage Tanks & Facilities
            </div>
        """)
    if '#FFA500' in used_colors:
        legend_items.append("""
            <div class="legend-item">
                <div class="legend-color" style="background-color: #FFA500;"></div>
                OWTS, MS4s, Landfills
            </div>
        """)
    if '#9370DB' in used_colors:
        legend_items.append("""
            <div class="legend-item">
                <div class="legend-color" style="background-color: #9370DB;"></div>
                Water Reuse & Resource Recovery
            </div>
        """)
    if '#000000' in used_colors:
        legend_items.append("""
            <div class="legend-item">
                <div class="legend-color" style="background-color: #000000;"></div>
                Other
            </div>
        """)

    cyto_html = f"""
    <html>
        <head>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js"></script>
            <style>
                #cy {{
                    width: 100%;
                    height: 600px;
                    display: block;
                    background-color: white;
                    position: absolute;
                }}
                body {{
                    margin: 0;
                    padding: 0;
                    background-color: white;
                    overflow: hidden;
                }}
                #legend {{
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    background: rgba(255, 255, 255, 0.9);
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                }}
                .legend-item {{
                    margin: 5px 0;
                }}
                .legend-color {{
                    display: inline-block;
                    width: 20px;
                    height: 20px;
                    margin-right: 5px;
                    vertical-align: middle;
                }}
            </style>
        </head>
        <body>
            <div id="cy"></div>
            <div id="legend">
                {''.join(legend_items)}
            </div>
            <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    var cy = cytoscape({{
                        container: document.getElementById('cy'),
                        elements: {elements},
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
                                    'line-color': '#999',
                                    'curve-style': 'bezier',
                                    'target-arrow-shape': 'triangle'
                                }}
                            }}
                        ],
                        layout: {{
                            name: 'cose',
                            padding: 10,
                            animate: false,
                            randomize: true,
                            componentSpacing: 4000,
                            nodeOverlap: 2,
                            nodeRepulsion: 500000,
                            idealEdgeLength: 10
                        }},
                        minZoom: 0.2,
                        maxZoom: 3
                    }});
                }});
            </script>
        </body>
    </html>
    """
    return cyto_html

# Load sewershed map
with open('processed_data/sewershed_map.pkl', 'rb') as f:
    sewershed_map = pickle.load(f)

st.title("U.S. Sewershed Network Visualization")
st.markdown("### Generate U.S. sewershed maps")

# Get states and counties
states = sorted(list(set([name.split(' - ')[0] for name in sewershed_map.keys() if ' - ' in name and name.split(' - ')[0] != 'Unspecified'])))
states.insert(0, "All States")

# Filters in a single row with buffer columns
buffer1, col1, col2, col3, buffer2 = st.columns([1,3,3,3,1])
with col1:
    selected_state = st.selectbox("Select state:", states, key="state_select")
with col2:
    counties = []
    if selected_state != "All States":
        counties = sorted(list(set([name.split(' - ')[1].split(' County Sewershed')[0] 
                                  for name in sewershed_map.keys() 
                                  if ' - ' in name and name.split(' - ')[0] == selected_state])))
        counties.insert(0, "All Counties")
        selected_county = st.selectbox("Select county:", counties, key="county_select")
    else:
        selected_county = None
with col3:
    keyword = st.text_input('Filter by facility name:', key="keyword_input")

# Filter results
results_list = []
for sewershed_id in sewershed_map.keys():
    if ' - ' not in sewershed_id:
        continue
        
    state = sewershed_id.split(' - ')[0]
    county = sewershed_id.split(' - ')[1].split(' County Sewershed')[0]
    
    state_match = selected_state == "All States" or state == selected_state
    county_match = not selected_county or selected_county == "All Counties" or county == selected_county
    
    keyword_match = True
    if keyword:
        facility_names = facilities.loc[facilities['CWNS_ID'].isin(sewershed_map[sewershed_id]['nodes']), 'FACILITY_NAME']
        keyword_match = facility_names.str.contains(keyword, case=False, na=False).any()
    
    if state_match and county_match and keyword_match:
        results_list.append(sewershed_id)

buffer3, col4, buffer4 = st.columns([1,9,1])
with col4:
    dropdown = st.selectbox("Select a sewershed:", sorted(results_list) if results_list else ["No matching sewersheds"])

if dropdown != "No matching sewersheds":
    try:
        html_plot = plot_sewershed(dropdown, sewershed_map, facilities)
        components.html(html_plot, height=600, scrolling=False)
    except Exception as e:
        st.error(f"Error plotting sewershed: {e}")

st.markdown("""
This tool visualizes facility connection system and treatment plant connections as described in the 2022 Clean Watersheds Needs Survey dataset. 
Data was downloaded from the "[Nationwide 2022 CWNS Dataset](https://sdwis.epa.gov/ords/sfdw_pub/r/sfdw/cwns_pub/data-download)".

This tool should be used for guidance only, and may not reflect the most recent or accurate depictions of any particular sewershed. 
For the most up-to-date information, see state-specific databases (e.g. CA State Water Boards) or the US EPA website.
""")