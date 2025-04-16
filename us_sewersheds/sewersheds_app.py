import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import streamlit.components.v1 as components
import dash_cytoscape as cyto
cyto.load_extra_layouts()
from dash import Dash, html
app = Dash()
server=app.server

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
facilities = pd.read_csv('processed_data/facilities_2022_merged.csv')[['CWNS_ID', 'DUMMY_ID', 'FACILITY_NAME','PERMIT_NUMBER','TOTAL_RES_POPULATION_2022','CURRENT_DESIGN_FLOW','FACILITY_TYPE']]

def add_newlines(text, max_length=20):
    """Add newlines after spaces for text longer than max_length"""
    if len(text) <= max_length:
        return text
        
    space_pos = text.find(' ', max_length)
    if space_pos == -1:
        return text
        
    return text[:space_pos] + '\n' + add_newlines(text[space_pos+1:], max_length)

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
    # print(nodes)
    connections = sewershed_map[sewershed_id]['connections']
    # print(connections)
    elements = []
    
    used_colors = set()
    
    # Find max population in network
    max_pop = 0
    for node in nodes:
        facility_mask = facilities['DUMMY_ID'] == node
        if not facilities[facility_mask].empty:
            population = facilities.loc[facility_mask, 'TOTAL_RES_POPULATION_2022'].iloc[0]
            if pd.notna(population) and population > max_pop:
                max_pop = population
    
    for node in nodes:
        facility_mask = facilities['DUMMY_ID'] == node
        if not facilities[facility_mask].empty:
            name = facilities.loc[facility_mask, 'FACILITY_NAME'].iloc[0]
            population = facilities.loc[facility_mask, 'TOTAL_RES_POPULATION_2022'].iloc[0]
            flow = facilities.loc[facility_mask, 'CURRENT_DESIGN_FLOW'].iloc[0] if pd.notna(facilities.loc[facility_mask, 'CURRENT_DESIGN_FLOW'].iloc[0]) else None
            permit_number = facilities.loc[facility_mask, 'PERMIT_NUMBER'].iloc[0]
            dummy_id = facilities.loc[facility_mask, 'DUMMY_ID'].iloc[0]
    
            # Add newlines after spaces after every 16 chars
            if len(name) > 20:
                name = add_newlines(name)
            facility_type = facilities.loc[facility_mask, 'FACILITY_TYPE'].iloc[0]
            color = sewershed_map[sewershed_id]['node_data'][node]['color']
            used_colors.add(color)
            shape = 'diamond' if facility_type and 'collection' in facility_type.lower() else 'ellipse' # different shape for collection
        else:
            name = str(node)
            population = 0
            flow = None
            color = sewershed_map[sewershed_id]['node_data'][node]['color']
            used_colors.add(color)
            shape = 'ellipse'
            
        elements.append({
            'data': {
                'id': str(node),
                'label': name,
                'color': color,
                'shape': shape,
                'TOTAL_RES_POPULATION_2022': str(int(population)) if pd.notna(population) else 'N/A',
                'CURRENT_DESIGN_FLOW': str(int(flow)) if flow is not None else 'N/A',
                'PERMIT_NUMBER': str(permit_number) if 'permit_number' in locals() and pd.notna(permit_number) else 'N/A',
                'DUMMY_ID': dummy_id if 'dummy_id' in locals() else node
            }
        })

    for conn in connections:
        elements.append({
            'data': {
                'source': str(conn[0]), 
                'target': str(conn[1]),
                'label': f'{conn[2]}%'  # Add flow percentage label
            }
        })

    # Build legend items based on used colors
    legend_items = []
    if '#ADD8E6' in used_colors:
        legend_items.append("""
            <div class="legend-item">
                <div class="legend-color" style="background-color: #C2E1EF;"></div>
                Centralized Treatment
            </div>
        """)
    if '#C4A484' in used_colors:
        legend_items.append("""
            <div class="legend-item">
                <div class="legend-color" style="background-color: #D4BCA0;"></div>
                Collection
            </div>
        """)
    if '#808080' in used_colors:
        legend_items.append("""
            <div class="legend-item">
                <div class="legend-color" style="background-color: #A0A0A0;"></div>
                Storage Tanks & Facilities
            </div>
        """)
    if '#FFA500' in used_colors:
        legend_items.append("""
            <div class="legend-item">
                <div class="legend-color" style="background-color: #FFBE4D;"></div>
                OWTS, MS4s, Landfills
            </div>
        """)
    if '#9370DB' in used_colors:
        legend_items.append("""
            <div class="legend-item">
                <div class="legend-color" style="background-color: #B095E6;"></div>
                Water Reuse & Resource Recovery
            </div>
        """)
    if '#FFFFC5' in used_colors:
        legend_items.append("""
            <div class="legend-item">
                <div class="legend-color" style="background-color: #FFFFD8;"></div>
                Other
            </div>
        """)
    if '#90EE90' in used_colors:
        legend_items.append("""
            <div class="legend-item">
                <div class="legend-color" style="background-color: #B0F5B0;"></div>
                Outfall
            </div>
        """)

    cyto_html = f"""

    <html>
        <head>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
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
                #info-display {{
                    position: absolute;
                    bottom: 10px;
                    left: 10px;
                    background: rgba(255, 255, 255, 0.9);
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    display: none;
                }}
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
                        var info = 'Upstream population served: ' + node.data('TOTAL_RES_POPULATION_2022');
                        if (node.data('PERMIT_NUMBER') && node.data('PERMIT_NUMBER') !== 'NA') {{
                            // Remove brackets if present in permit number
                            var permitNumber = node.data('PERMIT_NUMBER');
                            if (permitNumber.startsWith('[') && permitNumber.endsWith(']')) {{
                                permitNumber = permitNumber.substring(1, permitNumber.length - 1);
                            }}
                            info = 'Permit Number: ' + permitNumber + '<br>' + info;
                        }}
                        if (node.data('CURRENT_DESIGN_FLOW') && node.data('CURRENT_DESIGN_FLOW') !== 'NA') {{
                            info = 'Current Design Flow: ' + node.data('CURRENT_DESIGN_FLOW') + ' MGD' + '<br>' + info;
                        }}
                        infoDisplay.innerHTML = info;
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
    keyword = st.text_input('Filter by facility name or permit number:', key="keyword_input")

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
        facility_mask = facilities['DUMMY_ID'].isin(sewershed_map[sewershed_id]['nodes'])
        facility_names = facilities.loc[facility_mask, 'FACILITY_NAME']
        permit_numbers = facilities.loc[facility_mask, 'PERMIT_NUMBER']
        name_match = facility_names.str.contains(keyword, case=False, na=False).any()
        permit_match = permit_numbers.str.contains(keyword, case=False, na=False).any()
        keyword_match = name_match or permit_match
    
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
This tool visualizes sewers, treatment facilities, outfalls, and connections as described in the 2022 Clean Watersheds Needs Survey dataset. 
Data was downloaded from the "[Nationwide 2022 CWNS Dataset](https://sdwis.epa.gov/ords/sfdw_pub/r/sfdw/cwns_pub/data-download)".

This tool should be used for approximation and guidance only, and may not reflect the most recent or accurate depictions of any particular sewershed. 
For the most up-to-date information, confirm with local or state authorities.
""")