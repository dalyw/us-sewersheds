import pandas as pd
import numpy as np
import pickle

# IMPORT DATA
cwns_2012 = pd.read_csv('data/cwns/2012/Facility_Details.csv')
facility_permit = pd.read_csv('data/cwns/2022CWNS_NATIONAL_APR2024/FACILITY_PERMIT.csv', encoding='latin1', low_memory=False)
facilities = pd.read_csv('data/cwns/2022CWNS_NATIONAL_APR2024/FACILITIES.csv', encoding='latin1', low_memory=False)[['CWNS_ID', 'FACILITY_NAME']]
areas_county = pd.read_csv('data/cwns/2022CWNS_NATIONAL_APR2024/AREAS_COUNTY.csv', encoding='latin1', low_memory=False)[['CWNS_ID', 'COUNTY_NAME']]
facility_types = pd.read_csv('data/cwns/2022CWNS_NATIONAL_APR2024/FACILITY_TYPES.csv', encoding='latin1', low_memory=False)
discharges = pd.read_csv('data/cwns/2022CWNS_NATIONAL_APR2024/DISCHARGES.csv', encoding='latin1', low_memory=False)
population_wastewater = pd.read_csv('data/cwns/2022CWNS_NATIONAL_APR2024/POPULATION_WASTEWATER.csv', encoding='latin1', low_memory=False)
flow = pd.read_csv('data/cwns/2022CWNS_NATIONAL_APR2024/FLOW.csv', encoding='latin1', low_memory=False)
discharges['DISCHARGES_TO_CWNSID'] = pd.to_numeric(discharges['DISCHARGES_TO_CWNSID'], errors='coerce').astype('Int64')
discharges['CWNS_ID'] = pd.to_numeric(discharges['CWNS_ID'], errors='coerce').astype('Int64')
discharges = discharges.merge(facilities, on='CWNS_ID', how='left')
discharges = discharges.merge(areas_county, on='CWNS_ID', how='left')
pop_served_cwns = pd.concat([
    pd.read_csv('data/cwns/2022CWNS_NATIONAL_APR2024/POPULATION_WASTEWATER.csv', encoding='latin1', low_memory=False),
    pd.read_csv('data/cwns/2022CWNS_NATIONAL_APR2024/POPULATION_WASTEWATER_CONFIRMED.csv', encoding='latin1', low_memory=False),
    pd.read_csv('data/cwns/2022CWNS_NATIONAL_APR2024/POPULATION_DECENTRALIZED.csv', encoding='latin1', low_memory=False)
])

# Get total flow data
total_flow = flow[flow['FLOW_TYPE'] == 'Total Flow'][['CWNS_ID', 'CURRENT_DESIGN_FLOW']]

# save intermediate file for sewersheds_app.py
facilities[['CWNS_ID', 'FACILITY_NAME']].to_csv('processed_data/cwns_facilities_merged.csv', index=False)

# clean CWNS population data
print(f'{len(facilities)} CWNS facilities')
facilities = facilities.merge(facility_permit, on='CWNS_ID', how='left')
print(f'{len(facilities)} CWNS facilities after merging with facility_permit')
facilities = facilities.merge(facility_types, on='CWNS_ID', how='left')
print(f'{len(facilities)} CWNS facilities after merging with facility_types')
facilities = facilities.merge(total_flow, on='CWNS_ID', how='left')
print(f'{len(facilities)} CWNS facilities after merging with flow')
facilities_for_marimo = facilities.copy().merge(population_wastewater, on='CWNS_ID', how='left')
print(f'{len(facilities_for_marimo)} CWNS facilities after merging with population_wastewater')
facilities_for_marimo.to_csv('processed_data/cwns_facilities_merged.csv', index=False)

facilities = facilities[~facilities['FACILITY_NAME'].str.contains('stormwater', case=False)]
print(f'{len(facilities)} CWNS facilities after dropping stormwater in name')
patterns_to_remove = [r'WDR ', r'WDR-', r'WDR', r'Order WQ ', r'WDR Order No. ', r'Order No. ', r'Order ', r'NO. ', r'ORDER NO. ', r'NO.', r'ORDER ', r'DWQ- ', r'NO.·', r'. ']
replacements = {r'·': '-', r'\?': '-'}
facilities['PERMIT_NUMBER_cwns_clean'] = facilities['PERMIT_NUMBER'].astype(str).replace('|'.join(patterns_to_remove), '', regex=True)
for old, new in replacements.items():
    facilities['PERMIT_NUMBER_cwns_clean'] = facilities['PERMIT_NUMBER_cwns_clean'].str.replace(old, new, regex=True)
facilities = facilities[facilities['PERMIT_NUMBER'] != '2006-0003-DWQ']
print(f'{len(facilities)} CWNS facilities after dropping 2006-0003-DWQ from PERMIT_NUMBER')

# POPULATION
cwns_facilities = facilities.merge(pop_served_cwns, on='CWNS_ID', how='left').dropna(subset=['PERMIT_NUMBER'])
cwns_facilities = cwns_facilities.groupby(['CWNS_ID', 'PERMIT_NUMBER_cwns_clean'], as_index=False).agg({
    'TOTAL_RES_POPULATION_2022': 'sum',
    'TOTAL_RES_POPULATION_2042': 'sum',
    'CURRENT_DESIGN_FLOW': 'first',
    **{col: 'first' for col in cwns_facilities.columns if col not in ['CWNS_ID', 'PERMIT_NUMBER_cwns_clean', 
                                                                      'TOTAL_RES_POPULATION_2022', 'TOTAL_RES_POPULATION_2042',
                                                                      'CURRENT_DESIGN_FLOW']}
})
print(f'{len(cwns_facilities)} CWNS facilities after merging with pop served and cleaning')

# MERGE ON CWNS ID for 2012 data
cwns_2012_mapping = cwns_2012[['CWNS Number', 'Present Residential Total Receiving Treatment Population']].drop_duplicates(subset='CWNS Number')
cwns_2012_mapping.columns = ['CWNS_ID', 'TOTAL_RES_POPULATION_2012']
cwns_facilities = cwns_facilities.merge(cwns_2012_mapping, on='CWNS_ID', how='left')

def add_connection(row):
    connection = [row['CWNS_ID'], row['DISCHARGES_TO_CWNSID'], row['PRESENT_DISCHARGE_PERCENTAGE']]
    return connection

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
state_county_used = {}

for _sewershed_info in sewershed_map.values():
    # Get state and county info for nodes in this sewershed
    node_info = discharges.loc[
        discharges["CWNS_ID"].isin(_sewershed_info["nodes"]), 
        ["STATE_CODE", "COUNTY_NAME"]
    ]
    
    # Get most common state and county
    if len(node_info) > 0:
        state_counts = node_info["STATE_CODE"].value_counts()
        primary_state = state_counts.index[0] if len(state_counts) > 0 else "Unspecified"
        
        county_counts = node_info[
            node_info["STATE_CODE"] == primary_state
        ]["COUNTY_NAME"].value_counts()
        primary_county = county_counts.index[0] if len(county_counts) > 0 else "Unspecified"
    else:
        primary_state = "Unspecified"
        primary_county = "Unspecified"

    # Handle NaN values
    primary_state = "Unspecified" if pd.isna(primary_state) else primary_state
    primary_county = "Unspecified" if pd.isna(primary_county) else primary_county
    
    # Create state-county key for counting
    state_county_key = f"{primary_state}_{primary_county}"
    state_county_used[state_county_key] = state_county_used.get(state_county_key, 0) + 1
    
    # Create new name with state and county
    new_name = f"{primary_state} - {primary_county} County Sewershed {state_county_used[state_county_key]}"

    # Find center node based on connection counts
    connection_counts = {}
    for node in _sewershed_info["nodes"]:
        count = 0
        for connection in _sewershed_info["connections"]:
            if connection[0] == node or connection[1] == node:
                count += 1
        connection_counts[node] = count
    center = max(connection_counts.items(), key=lambda x: x[1])[0]
    
    _sewershed_info["center"] = center
    
    # Add flow and population data for each node
    node_data = {}
    for node in _sewershed_info["nodes"]:
        node_data[node] = {
            'flow': cwns_facilities.loc[cwns_facilities['CWNS_ID'] == node, 'CURRENT_DESIGN_FLOW'].iloc[0] if not cwns_facilities[cwns_facilities['CWNS_ID'] == node].empty else None,
            'population': cwns_facilities.loc[cwns_facilities['CWNS_ID'] == node, 'TOTAL_RES_POPULATION_2022'].iloc[0] if not cwns_facilities[cwns_facilities['CWNS_ID'] == node].empty else None
        }
    _sewershed_info["node_data"] = node_data
    
    new_sewershed_map[new_name] = _sewershed_info

sewershed_map = new_sewershed_map

# save sewershed map to pickle file
pickle.dump(sewershed_map, open('processed_data/sewershed_map.pkl', 'wb'))