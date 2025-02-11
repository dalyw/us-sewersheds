import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import streamlit.components.v1 as components
import dash_cytoscape as cyto
from dash import Dash, html

# Set page config
st.set_page_config(page_title="CA Sewersheds", layout="wide", initial_sidebar_state="expanded")

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

    # Create elements list for Cytoscape
    elements = []
    
    # Add nodes
    for node in nodes:
        facility_mask = facilities['CWNS_ID'] == node
        if not facilities[facility_mask].empty:
            name = facilities.loc[facility_mask, 'FACILITY_NAME'].iloc[0]
        else:
            name = str(node)
            
        elements.append({
            'data': {
                'id': str(node),
                'label': name
            }
        })

    # Add edges
    for conn in connections:
        elements.append({
            'data': {
                'source': str(conn[0]),
                'target': str(conn[1])
            }
        })

    # Create Cytoscape component HTML with full JavaScript dependencies
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
            </style>
        </head>
        <body>
            <div id="cy"></div>
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
                                    'background-color': '#666',
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
                            padding: 50,
                            animate: false,
                            randomize: true,
                            componentSpacing: 100,
                            nodeOverlap: 20
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

def add_connection(row):
    connection = [row['CWNS_ID'], row['DISCHARGES_TO_CWNSID'], row['PRESENT_DISCHARGE_PERCENTAGE']]
    return connection

# Load pre-built sewershed map from pickle file
with open('processed_data/sewershed_map.pkl', 'rb') as f:
    sewershed_map = pickle.load(f)

st.markdown("""
### Generate U.S. sewershed maps
""")

# Get unique states and counties from sewershed names
states = sorted(list(set([name.split(' - ')[0] for name in sewershed_map.keys() if ' - ' in name and name.split(' - ')[0] != 'Unspecified'])))
states.insert(0, "All States")

# Add state selection
selected_state = st.selectbox("Select a state:", states)

# Get counties for selected state
counties = []
if selected_state != "All States":
    counties = sorted(list(set([
        name.split(' - ')[1].split(' County Sewershed')[0]
        for name in sewershed_map.keys() 
        if ' - ' in name and name.split(' - ')[0] == selected_state
    ])))
    counties.insert(0, "All Counties")

# Add county selection if state is selected
if selected_state != "All States":
    selected_county = st.selectbox("Select a county:", counties)
else:
    selected_county = None

# Add search functionality
keyword = st.text_input('Filter by facility name: ')

# Filter results based on selections
results_list = []
for sewershed_id in sewershed_map.keys():
    if ' - ' not in sewershed_id:
        continue
        
    state = sewershed_id.split(' - ')[0]
    county = sewershed_id.split(' - ')[1].split(' County Sewershed')[0]
    
    # Check if sewershed matches state/county filters
    state_match = selected_state == "All States" or state == selected_state
    county_match = not selected_county or selected_county == "All Counties" or county == selected_county
    
    # Check if sewershed matches keyword search
    keyword_match = True
    if keyword:
        facility_names = facilities.loc[facilities['CWNS_ID'].isin(sewershed_map[sewershed_id]['nodes']), 'FACILITY_NAME']
        keyword_match = facility_names.str.contains(keyword, case=False, na=False).any()
    
    if state_match and county_match and keyword_match:
        results_list.append(sewershed_id)

# Add dropdown for sewershed selection
dropdown = st.selectbox("Select a sewershed:", sorted(results_list) if results_list else ["No matching sewersheds"])

# Display plot directly in Streamlit when sewershed is selected
if dropdown != "No matching sewersheds":
    try:
        html_plot = plot_sewershed(dropdown, sewershed_map, facilities)
        components.html(html_plot, height=600, scrolling=False)
    except Exception as e:
        st.error(f"Error plotting sewershed: {e}")
        print(f"Error plotting sewershed: {e}")

# Add markdown description
st.markdown(
    r"""
    This tool visualizes facility connection system and treatment plant connections as described in the 2022 Clean Watersheds Needs Survey dataset. 
    Data was downloaded from the "[Nationwide 2022 CWNS Dataset](https://sdwis.epa.gov/ords/sfdw_pub/r/sfdw/cwns_pub/data-download)".

    This tool should be used for guidance only, and may not reflect the most recent or accurate depictions of any particular sewershed. 
    For the most up-to-date information, see state-specific databases (e.g. CA State Water Boards) or the US EPA website.
    """
)