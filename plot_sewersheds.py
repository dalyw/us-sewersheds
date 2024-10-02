import marimo

__generated_with = "0.8.15"
app = marimo.App()


@app.cell
def __():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx
    from helper_functions import read_dmr
    import pickle
    return np, nx, pd, pickle, plt, read_dmr


@app.cell
def __(mo):
    mo.md(
        r"""
        This tool visualizes facility connection system and treatment plant connections as described in the 2022 Clean Watersheds Needs Suvey dataset. Data was downloaded from the "[2022 CWNS Dataset by State](https://sdwis.epa.gov/ords/sfdw_pub/r/sfdw/cwns_pub/data-download)" for California.

        This tool should be used for guidance only, and may not reflect the most recent or accurate depictions of any particular California sewershed. For the most up-to-date information, see the [CA State Water Boards database](https://www.waterboards.ca.gov/ciwqs/).
        """
    )
    return


@app.cell
def __(pd, pickle):
    sewershed_map = pickle.load(open('processed_data/step2/sewershed_map.pkl', 'rb'))
    facilities = pd.read_csv('processed_data/step2/cwns_facilities_merged.csv')
    return facilities, sewershed_map


@app.cell
def __(nx, plt):
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
        facility_names = {node: f'{(facilities.loc[facilities['CWNS_ID'] == node, 'FACILITY_NAME'].iloc[0] if not facilities[facilities['CWNS_ID'] == node].empty else str(node))}'.replace('(', '\n(', 1) if len(f'{(facilities.loc[facilities['CWNS_ID'] == node, 'FACILITY_NAME'].iloc[0] if not facilities[facilities['CWNS_ID'] == node].empty else str(node))}') > 20 else f'{(facilities.loc[facilities['CWNS_ID'] == node, 'FACILITY_NAME'].iloc[0] if not facilities[facilities['CWNS_ID'] == node].empty else str(node))}' for node in nodes}
        facility_permit_numbers = {node: f'{(facilities.loc[facilities['CWNS_ID'] == node, 'PERMIT_NUMBER'].iloc[0] if not facilities[facilities['CWNS_ID'] == node].empty else 'N/A')}' for node in nodes}
        facility_pop = {node: f'{(facilities.loc[facilities['CWNS_ID'] == node, 'TOTAL_RES_POPULATION_2022'].iloc[0] if not facilities[facilities['CWNS_ID'] == node].empty else 'N/A')}' for node in nodes}
        node_labels = {node: f'{facility_names[node]}\nPermit: {facility_permit_numbers[node]}\nPop. 2022: {int(float(facility_pop[node])) if facility_pop[node] != "N/A" else facility_pop[node]}' for node in nodes}
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

        plt.figure(figsize=(6+4*len(nodes)/10, 4+3*len(nodes)/10))
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
        return plt, G, pos
    return plot_sewershed,


@app.cell
def __(mo):
    mo.md(
        """
        \n    ### Display the sewershed\n
        \n    ##### You may try re-generationg the graphic a few times to get an improved layout.\n
        """
    )
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
