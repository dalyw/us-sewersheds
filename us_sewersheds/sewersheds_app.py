import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle

# Set page config
st.set_page_config(page_title="CA Sewersheds", layout="wide")

# Add markdown description
st.markdown(
    r"""
    This tool visualizes facility connection system and treatment plant connections as described in the 2022 Clean Watersheds Needs Survey dataset. 
    Data was downloaded from the "[Nationwide 2022 CWNS Dataset](https://sdwis.epa.gov/ords/sfdw_pub/r/sfdw/cwns_pub/data-download)".

    This tool should be used for guidance only, and may not reflect the most recent or accurate depictions of any particular sewershed. 
    For the most up-to-date information, see state-specific databases (e.g. [CA State Water Boards](https://www.waterboards.ca.gov/ciwqs/) or the US EPA website.
    """
)

# Load data
facilities = pd.read_csv('processed_data/cwns_facilities_merged.csv')[['CWNS_ID', 'FACILITY_NAME','PERMIT_NUMBER','TOTAL_RES_POPULATION_2022','FACILITY_TYPE']]

# DEFINE PLOTTING FUNCTIONS
def plot_sewershed(sewershed_id, sewershed_map, facilities):
    """
    Each entry in sewershed_map is a dictionary with keys 'nodes' and 'connections'. 
    Plots a directed graph of a given sewershed

    Inputs:
    - sewershed_id: string, the ID of the sewershed to plot
    - sewershed_map: dictionary, the sewershed map
    - facilities: pandas dataframe, the facilities dataframe

    Outputs:
    - None
    """
    G = nx.DiGraph()
    nodes = sewershed_map[sewershed_id]['nodes']
    connections = sewershed_map[sewershed_id]['connections']
    center_node = sewershed_map[sewershed_id]['center']
    facility_names = {}
    for node in nodes:
        # Get facility name from dataframe or use node ID if not found
        facility_mask = facilities['CWNS_ID'] == node
        if not facilities[facility_mask].empty:
            name = facilities.loc[facility_mask, 'FACILITY_NAME'].iloc[0]
        else:
            name = str(node)
            
        # Add line break before parentheses if name is long
        if len(name) > 20:
            name = name.replace('(', '\n(', 1)
            
        facility_names[node] = name
    facility_permit_numbers = {node: f"{(facilities.loc[facilities['CWNS_ID'] == node, 'PERMIT_NUMBER'].iloc[0] if not facilities[facilities['CWNS_ID'] == node].empty else 'N/A')}" for node in nodes}
    facility_pop = {node: f"{(facilities.loc[facilities['CWNS_ID'] == node, 'TOTAL_RES_POPULATION_2022'].iloc[0] if not facilities[facilities['CWNS_ID'] == node].empty else 'N/A')}" for node in nodes}
    node_labels = {node: f"{facility_names[node]}\nPermit: {facility_permit_numbers[node]}\nPop. 2022: {int(float(facility_pop[node])) if facility_pop[node] != 'N/A' else facility_pop[node]}" for node in nodes}
    G.add_nodes_from(nodes)
    G.add_edges_from([(conn[0], conn[1], {'label': f'{conn[2]}%'}) for conn in connections])
    
    # Use a spring layout as a starting point
    pos = nx.spring_layout(G)

    # Function to calculate repulsive forces between nodes
    def repulsive_force(pos1, pos2, k=0.1):
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        distance = max((dx**2 + dy**2)**0.5, 0.01)
        force = k / distance**2
        return force * dx / distance, force * dy / distance

    # Function to calculate attractive forces for connected nodes
    def attractive_force(pos1, pos2, k=0.01):
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        distance = max((dx**2 + dy**2)**0.5, 0.01)
        force = k * distance
        return force * dx / distance, force * dy / distance

    # Iteratively adjust positions
    iterations = 50
    for _ in range(iterations):
        new_pos = pos.copy()
        for node in G.nodes():
            fx, fy = 0, 0
            for other in G.nodes():
                if node != other:
                    dfx, dfy = repulsive_force(pos[node], pos[other])
                    fx += dfx
                    fy += dfy
            for neighbor in G.neighbors(node):
                dfx, dfy = attractive_force(pos[node], pos[neighbor])
                fx += dfx
                fy += dfy
            new_pos[node] = (pos[node][0] + fx, pos[node][1] + fy)
        pos = new_pos

    # Normalize positions
    x_values, y_values = zip(*pos.values())
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    for node in pos:
        x = (pos[node][0] - x_min) / (x_max - x_min) if x_max > x_min else 0.5
        y = (pos[node][1] - y_min) / (y_max - y_min) if y_max > y_min else 0.5
        pos[node] = (x, y)

    # Add buffer around edges
    buffer = 0.1
    for node in pos:
        x, y = pos[node]
        x = max(buffer, min(1 - buffer, x))
        y = max(buffer, min(1 - buffer, y))
        pos[node] = (x, y)

    # Ensure minimum distance between nodes
    min_dist = 0.2
    for _ in range(20):
        for node1 in G.nodes():
            for node2 in G.nodes():
                if node1 != node2:
                    dx = pos[node2][0] - pos[node1][0]
                    dy = pos[node2][1] - pos[node1][1]
                    dist = max((dx**2 + dy**2)**0.5, 0.01)
                    if dist < min_dist:
                        move = (min_dist - dist) / 2
                        pos[node1] = (pos[node1][0] - move*dx/dist, pos[node1][1] - move*dy/dist)
                        pos[node2] = (pos[node2][0] + move*dx/dist, pos[node2][1] + move*dy/dist)

    fig = plt.figure(figsize=(6+4*len(nodes)/10, 4+3*len(nodes)/10))
    ax1 = plt.subplot(111)
    ax1.margins(0.5*(3/len(nodes))) 
    nx.draw_networkx_edge_labels(G, pos, ax=ax1, edge_labels=nx.get_edge_attributes(G, 'label'))

    node_shapes = {node: 's' if any('Treatment' in facility_type for facility_type in facilities.loc[facilities['CWNS_ID'] == node, 'FACILITY_TYPE']) else '2' for node in nodes}
    for shape in set(node_shapes.values()):
        node_list = [node for node in nodes if node_shapes[node] == shape]
        nx.draw_networkx_nodes(G, pos, ax=ax1, nodelist=node_list, node_shape=shape, node_size=3000, node_color='lightblue', alpha = 0.7)

    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='gray', alpha=0.7, arrows=True, arrowsize=20, arrowstyle='->', min_source_margin=30, min_target_margin=30)
    nx.draw_networkx_labels(G, pos, ax=ax1, labels=node_labels, font_size=10)

    plt.title(f'{sewershed_id}')
    plt.axis('off')
    plt.tight_layout()
    return fig, G, pos

def add_connection(row):
    connection = [row['CWNS_ID'], row['DISCHARGES_TO_CWNSID'], row['PRESENT_DISCHARGE_PERCENTAGE']]
    return connection

# Load pre-built sewershed map from pickle file
with open('processed_data/sewershed_map.pkl', 'rb') as f:
    sewershed_map = pickle.load(f)

st.markdown("""
### Display the sewershed

##### You may try re-generating the graphic a few times to get an improved layout.
""")
# Get unique states and counties from sewershed names
print(sewershed_map.keys())
states = sorted(list(set([name.split(' - ')[0] for name in sewershed_map.keys() if ' - ' in name and name.split(' - ')[0] != 'Unspecified'])))
print(states)
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

# Add button to generate plot
if st.button('Generate Sewershed Plot') and dropdown != "No matching sewersheds":
    fig, G, pos = plot_sewershed(dropdown, sewershed_map, facilities)
    st.pyplot(fig)