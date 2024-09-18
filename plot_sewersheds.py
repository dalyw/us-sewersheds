import marimo

__generated_with = "0.8.15"
app = marimo.App()


@app.cell
def __():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx
    return np, nx, pd, plt


@app.cell
def __(nx, plt):
    # DEFINE PLOTTING FUNCTIONS

    def add_connection(row):
        connection = [row['CWNS_ID'], row['DISCHARGES_TO_CWNSID'], row['PRESENT_DISCHARGE_PERCENTAGE']]
        return connection

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
        facility_names = {node: f'{(facilities.loc[facilities['CWNS_ID'] == node, 'FACILITY_NAME'].iloc[0] if not facilities[facilities['CWNS_ID'] == node].empty else str(node))}' for node in nodes}
        facility_permit_numbers = {node: f'{(facilities.loc[facilities['CWNS_ID'] == node, 'PERMIT_NUMBER'].iloc[0] if not facilities[facilities['CWNS_ID'] == node].empty else str(node))}' for node in nodes}
        node_labels = {node: f'{facility_names[node]}\n({facility_permit_numbers[node]})' for node in nodes}
        G.add_nodes_from(nodes)
        G.add_edges_from([(conn[0], conn[1], {'label': f'{conn[2]}%'}) for conn in connections])
        # pos = nx.spring_layout(G)
        pos = nx.spring_layout(G, k=3, fixed={center_node: (0, 0)}, pos={center_node: (0, 0)})
        plt.figure(figsize=(9, 6))
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'))
        nx.draw(G, pos, labels=node_labels, with_labels=True, node_size=3000, node_color='lightblue', font_size=6, font_weight='bold', edge_color='gray', alpha=0.7, arrows=True)
        plt.title(f'{sewershed_id}')
        plt.axis('off')
        plt.tight_layout()
        return plt
    return add_connection, plot_sewershed


@app.cell
def __(pd):
    # IMPORT DATA

    facility_permit = pd.read_csv('CA_2022CWNS_APR2024/FACILITY_PERMIT.csv', encoding='latin1', low_memory=False)
    facilities = pd.read_csv('CA_2022CWNS_APR2024/FACILITIES.csv', encoding='latin1', low_memory=False)
    areas_county = pd.read_csv('CA_2022CWNS_APR2024/AREAS_COUNTY.csv', encoding='latin1', low_memory=False)
    discharges = pd.read_csv('CA_2022CWNS_APR2024/DISCHARGES.csv', encoding='latin1', low_memory=False)
    discharges['DISCHARGES_TO_CWNSID'] = pd.to_numeric(discharges['DISCHARGES_TO_CWNSID'], errors='coerce').astype('Int64')
    discharges['CWNS_ID'] = pd.to_numeric(discharges['CWNS_ID'], errors='coerce').astype('Int64')
    facilities = facilities[['CWNS_ID', 'FACILITY_NAME']]
    facilities.set_index('CWNS_ID', inplace=True)
    discharges = discharges.merge(facilities, left_on='CWNS_ID', right_index=True, how='left')
    discharges = discharges.merge(areas_county, left_on='CWNS_ID', right_on='CWNS_ID', how='left')
    facilities = facilities.merge(facility_permit, left_on='CWNS_ID', right_on='CWNS_ID', how='left')
    return areas_county, discharges, facilities, facility_permit


@app.cell
def __(add_connection, discharges, pd):
    # BUILD SEWERSHED MAP

    sewershed_map = {}
    facilities_already_mapped = []
    for _, row in discharges[discharges['DISCHARGE_TYPE'] == 'Discharge To Another Facility'].iterrows():
        cwns_id, discharges_to = row['CWNS_ID'], row['DISCHARGES_TO_CWNSID']
        if cwns_id not in facilities_already_mapped and discharges_to not in facilities_already_mapped:
            new_sewershed_id = len(sewershed_map) + 1
            sewershed_map[new_sewershed_id] = {
                'nodes': [cwns_id, discharges_to],
                'connections': [add_connection(row)]
            }
            facilities_already_mapped.append(cwns_id)
            facilities_already_mapped.append(discharges_to)
        else:
            for sewershed_info in sewershed_map.values():
                if cwns_id in sewershed_info['nodes'] or discharges_to in sewershed_info['nodes']:
                    if cwns_id not in sewershed_info['nodes']:
                        sewershed_info['nodes'].append(cwns_id)
                    if discharges_to not in sewershed_info['nodes']:   
                        sewershed_info['nodes'].append(discharges_to)
                    sewershed_info['connections'].append(add_connection(row))
                    facilities_already_mapped.append(cwns_id)
                    facilities_already_mapped.append(discharges_to)
                    break
                    
    new_sewershed_map = {}
    name_used = {}
    for _sewershed_info in sewershed_map.values():
        county_names = discharges.loc[discharges['CWNS_ID'].isin(_sewershed_info['nodes']), 'COUNTY_NAME']
        primary_county = county_names.value_counts().index[0] if len(county_names.value_counts()) > 0 else 'Unspecified'
        primary_county = 'Unspecified' if pd.isna(primary_county) else primary_county
        name_used[primary_county] = name_used.get(primary_county, 0) + 1
        new_name = f'{primary_county} County Sewershed {name_used[primary_county]}'
        connection_counts = {}
        for node in _sewershed_info['nodes']:
            count = 0
            for connection in _sewershed_info['connections']:
                if connection[0] == node or connection[1] == node:
                    count += 1
            connection_counts[node] = count
        center = max(connection_counts.items(), key=lambda x: x[1])[0]
        _sewershed_info['center'] = center
        new_sewershed_map[new_name] = _sewershed_info
    sewershed_map = new_sewershed_map
    return (
        center,
        connection,
        connection_counts,
        count,
        county_names,
        cwns_id,
        discharges_to,
        facilities_already_mapped,
        name_used,
        new_name,
        new_sewershed_id,
        new_sewershed_map,
        node,
        primary_county,
        row,
        sewershed_info,
        sewershed_map,
    )


@app.cell
def __(mo):
    mo.md("""\n    ### Display the sewershed\n""")
    return


@app.cell
def __(mo):
    keyword = mo.ui.text(label='Filter facility name (or portion of name): ')
    keyword
    return keyword,


@app.cell
def __(facilities, keyword, sewershed_map):
    results_list = []

    for sewershed_id, _sewershed_info in sewershed_map.items():
        facility_names = facilities.loc[facilities['CWNS_ID'].isin(_sewershed_info['nodes']), 'FACILITY_NAME']
        if facility_names.str.contains(keyword.value, case=False, na=False).any():
            results_list.append(sewershed_id)
    return facility_names, results_list, sewershed_id


@app.cell
def __(mo, results_list):
    dropdown = mo.ui.dropdown(
        options=sorted(results_list), label="Select a sewershed: "
    )
    dropdown
    return dropdown,


@app.cell
def __(mo):
    button = mo.ui.run_button(label = 'Generate Plot')
    button
    return button,


@app.cell
def __(button, dropdown, facilities, mo, plot_sewershed, sewershed_map):
    mo.stop(not button.value)
    plot = plot_sewershed(dropdown.value, sewershed_map, facilities)
    plot.gca()
    return plot,


@app.cell
def __():
    return


@app.cell
def __():
    import marimo as mo
    return mo,


if __name__ == "__main__":
    app.run()
