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
        node_labels = {node: f'{facility_names[node]}\nPermit no: {facility_permit_numbers[node]}' for node in nodes}

        G.add_nodes_from(nodes)
        G.add_edges_from([(conn[0], conn[1], {'label': f'{conn[2]}%'}) for conn in connections])
        pos = nx.spring_layout(G, k=3, fixed={center_node: (0, 0)}, pos={center_node: (0, 0)})
        plt.figure(figsize=(9, 6))
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'))

        node_shapes = {node: 's' if any('Treatment' in facility_type for facility_type in facilities.loc[facilities['CWNS_ID'] == node, 'FACILITY_TYPE']) else '2' for node in nodes}
        for shape in set(node_shapes.values()):
            node_list = [node for node in nodes if node_shapes[node] == shape]
            nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_shape=shape, node_size=3000, node_color='lightblue', alpha = 0.7)

        
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.7, arrows=True, arrowsize=20, arrowstyle='->', min_source_margin=30, min_target_margin=30)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=6, font_weight='bold')

        plt.title(f'{sewershed_id}')
        plt.axis('off')
        plt.tight_layout()
        return plt, G, pos
    return add_connection, plot_sewershed


@app.cell
def __(pd):
    # IMPORT DATA

    facility_permit = pd.read_csv('CA_2022CWNS_APR2024/FACILITY_PERMIT.csv', encoding='latin1', low_memory=False)
    facilities = pd.read_csv('CA_2022CWNS_APR2024/FACILITIES.csv', encoding='latin1', low_memory=False)
    areas_county = pd.read_csv('CA_2022CWNS_APR2024/AREAS_COUNTY.csv', encoding='latin1', low_memory=False)
    facility_types = pd.read_csv('CA_2022CWNS_APR2024/FACILITY_TYPES.csv', encoding='latin1', low_memory=False)
    population_wastewater = pd.read_csv('CA_2022CWNS_APR2024/POPULATION_WASTEWATER.csv', encoding='latin1', low_memory=False)
    population_wastewater_confirmed = pd.read_csv('CA_2022CWNS_APR2024/POPULATION_WASTEWATER_CONFIRMED.csv', encoding='latin1', low_memory=False)
    population_decentralized = pd.read_csv('CA_2022CWNS_APR2024/POPULATION_DECENTRALIZED.csv', encoding='latin1', low_memory=False)
    discharges = pd.read_csv('CA_2022CWNS_APR2024/DISCHARGES.csv', encoding='latin1', low_memory=False)
    discharges['DISCHARGES_TO_CWNSID'] = pd.to_numeric(discharges['DISCHARGES_TO_CWNSID'], errors='coerce').astype('Int64')
    discharges['CWNS_ID'] = pd.to_numeric(discharges['CWNS_ID'], errors='coerce').astype('Int64')
    facilities = facilities[['CWNS_ID', 'FACILITY_NAME']]
    facilities.set_index('CWNS_ID', inplace=True)
    discharges = discharges.merge(facilities, left_on='CWNS_ID', right_index=True, how='left')
    discharges = discharges.merge(areas_county, left_on='CWNS_ID', right_on='CWNS_ID', how='left')
    facilities = facilities.merge(facility_permit, left_on='CWNS_ID', right_on='CWNS_ID', how='left')
    facilities = facilities.merge(facility_types, left_on='CWNS_ID', right_on='CWNS_ID', how='left')
    return (
        areas_county,
        discharges,
        facilities,
        facility_permit,
        facility_types,
        population_decentralized,
        population_wastewater,
        population_wastewater_confirmed,
    )


@app.cell
def __(add_connection, discharges, pd):
    # BUILD SEWERSHED MAP

    sewershed_map = {}
    rows_new_sewershed = []
    facilities_already_mapped = []
    for _, row in discharges[
        discharges["DISCHARGE_TYPE"] == "Discharge To Another Facility"
    ].iterrows():
        cwns_id, discharges_to = row["CWNS_ID"], row["DISCHARGES_TO_CWNSID"]
        if (
            cwns_id not in facilities_already_mapped
            and discharges_to not in facilities_already_mapped
        ):
            rows_new_sewershed.append(row.name)
            new_sewershed_id = len(sewershed_map) + 1
            sewershed_map[new_sewershed_id] = {
                "nodes": set([cwns_id, discharges_to]),
                "connections": [add_connection(row)],
            }
            facilities_already_mapped.extend([cwns_id, discharges_to])
        else:
            for sewershed_info in sewershed_map.values():
                if (
                    cwns_id in sewershed_info["nodes"]
                    or discharges_to in sewershed_info["nodes"]
                ):
                    sewershed_info["nodes"].update([cwns_id, discharges_to])
                    sewershed_info["connections"].append(add_connection(row))
                    facilities_already_mapped.extend([cwns_id, discharges_to])
                    break

    # Consolidate sewersheds with redundant nodes
    sewershed_ids = list(sewershed_map.keys())
    for i in range(len(sewershed_ids)):
        for j in range(i + 1, len(sewershed_ids)):
            id1, id2 = sewershed_ids[i], sewershed_ids[j]
            if id1 in sewershed_map and id2 in sewershed_map:
                if sewershed_map[id1]["nodes"] & sewershed_map[id2]["nodes"]:
                    # Merge sewersheds
                    sewershed_map[id1]["nodes"].update(sewershed_map[id2]["nodes"])
                    sewershed_map[id1]["connections"].extend(sewershed_map[id2]["connections"])
                    del sewershed_map[id2]


    new_sewershed_map = {}
    name_used = {}
    for _sewershed_info in sewershed_map.values():
        county_names = discharges.loc[
            discharges["CWNS_ID"].isin(_sewershed_info["nodes"]), "COUNTY_NAME"
        ]
        primary_county = (
            county_names.value_counts().index[0]
            if len(county_names.value_counts()) > 0
            else "Unspecified"
        )
        primary_county = (
            "Unspecified" if pd.isna(primary_county) else primary_county
        )
        name_used[primary_county] = name_used.get(primary_county, 0) + 1
        new_name = f"{primary_county} County Sewershed {name_used[primary_county]}"
        connection_counts = {}
        for node in _sewershed_info["nodes"]:
            count = 0
            for connection in _sewershed_info["connections"]:
                if connection[0] == node or connection[1] == node:
                    count += 1
            connection_counts[node] = count
        center = max(connection_counts.items(), key=lambda x: x[1])[0]
        _sewershed_info["center"] = center
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
        i,
        id1,
        id2,
        j,
        name_used,
        new_name,
        new_sewershed_id,
        new_sewershed_map,
        node,
        primary_county,
        row,
        rows_new_sewershed,
        sewershed_ids,
        sewershed_info,
        sewershed_map,
    )


@app.cell
def __(mo):
    mo.md("""\n    ### Display the sewershed\n""")
    return


@app.cell
def __(mo):
    keyword = mo.ui.text(label='Filter by facility name (or portion of name): ')
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
    button = mo.ui.run_button(label = 'Generate Sewershed Plot')
    button
    return button,


@app.cell
def __(button, dropdown, facilities, mo, plot_sewershed, sewershed_map):
    mo.stop(not button.value)
    plot, G, pos = plot_sewershed(dropdown.value, sewershed_map, facilities)
    plot.gca()
    return G, plot, pos


@app.cell
def __():
    import marimo as mo
    return mo,


if __name__ == "__main__":
    app.run()
