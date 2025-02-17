import pandas as pd
import numpy as np
import pickle
import copy

# IMPORT DATA
facilities_2012 = pd.read_csv('data/2012/Facility_Details.csv')
facilities_2022 = pd.read_csv('data/2022CWNS_NATIONAL_APR2024/FACILITIES.csv', encoding='latin1', low_memory=False)
facilities_2022 = facilities_2022[facilities_2022['STATE_CODE']=='CA'][['CWNS_ID', 'FACILITY_NAME']]
# facilities_2022 = facilities_2022[['CWNS_ID', 'FACILITY_NAME']]

facility_permit = pd.read_csv('data/2022CWNS_NATIONAL_APR2024/FACILITY_PERMIT.csv', encoding='latin1', low_memory=False)[['CWNS_ID', 'PERMIT_NUMBER']]
facility_permit = facility_permit.groupby('CWNS_ID')['PERMIT_NUMBER'].agg(list).reset_index()
facility_permit.name = 'facility_permit'

areas_county = pd.read_csv('data/2022CWNS_NATIONAL_APR2024/AREAS_COUNTY.csv', encoding='latin1', low_memory=False)
areas_county = areas_county[areas_county['COUNTY_PRIMARY_FLAG'] == 'Y'][['CWNS_ID', 'COUNTY_NAME']]
areas_county.name = 'areas_county'

facility_types = pd.read_csv('data/2022CWNS_NATIONAL_APR2024/FACILITY_TYPES.csv', encoding='latin1', low_memory=False)
facility_types = facility_types.drop('CHANGE_TYPE', axis=1).drop_duplicates()
facility_types.name = 'facility_types'

flow = pd.read_csv('data/2022CWNS_NATIONAL_APR2024/FLOW.csv', encoding='latin1', low_memory=False)[['CWNS_ID','FLOW_TYPE','CURRENT_DESIGN_FLOW']]
total_flow = flow[flow['FLOW_TYPE'] == 'Total Flow'][['CWNS_ID', 'CURRENT_DESIGN_FLOW']]
total_flow.name = 'total_flow'

# Read population data files
pop_columns = ['CWNS_ID','TOTAL_RES_POPULATION_2022', 'TOTAL_RES_POPULATION_2042']
# pop_columns2 = ['CWNS_ID',"RESIDENTIAL_POP_2022","RESIDENTIAL_POP_2042"]
pop_wastewater = pd.read_csv('data/2022CWNS_NATIONAL_APR2024/POPULATION_WASTEWATER.csv', encoding='latin1', low_memory=False)[pop_columns]
pop_wastewater_confirmed = pd.read_csv('data/2022CWNS_NATIONAL_APR2024/POPULATION_WASTEWATER_CONFIRMED.csv', encoding='latin1', low_memory=False)[pop_columns]
pop_decentralized = pd.read_csv('data/2022CWNS_NATIONAL_APR2024/POPULATION_DECENTRALIZED.csv', encoding='latin1', low_memory=False)[['CWNS_ID', 'RESIDENTIAL_POP_2022', 'RESIDENTIAL_POP_2042']].rename(columns={
    'RESIDENTIAL_POP_2022': 'TOTAL_RES_POPULATION_2022',
    'RESIDENTIAL_POP_2042': 'TOTAL_RES_POPULATION_2042'
})[pop_columns]
pop_served_cwns = pd.concat([
    pop_wastewater_confirmed,
    pop_wastewater,
    pop_decentralized
]).drop_duplicates(subset='CWNS_ID', keep='first')
pop_served_cwns.name = 'pop_served_cwns'

# merge CWNS data files
print(f'{len(facilities_2022)} CWNS facilities')
for df in [facility_permit, facility_types, total_flow, pop_served_cwns, areas_county]:
    facilities_2022 = facilities_2022.merge(df, on='CWNS_ID', how='left')
    print(f'{len(facilities_2022)} CWNS facilities after merging with {df.name}')

patterns_to_remove = [r'WDR ', r'WDR-', r'WDR', r'Order WQ ', r'WDR Order No. ', r'Order No. ', r'Order ', r'NO. ', r'ORDER NO. ', r'NO.', r'ORDER ', r'DWQ- ', r'NO.·', r'. ']
replacements = {r'·': '-', r'\?': '-'}
facilities_2022['PERMIT_NUMBER_cwns_clean'] = facilities_2022['PERMIT_NUMBER'].astype(str).replace('|'.join(patterns_to_remove), '', regex=True)
for old, new in replacements.items():
    facilities_2022['PERMIT_NUMBER_cwns_clean'] = facilities_2022['PERMIT_NUMBER_cwns_clean'].str.replace(old, new, regex=True)

# TODO: check if this is necessary
facilities_2022 = facilities_2022.groupby(['CWNS_ID', 'PERMIT_NUMBER_cwns_clean', 'FACILITY_TYPE'], as_index=False).agg({
    'TOTAL_RES_POPULATION_2022': 'sum',
    'TOTAL_RES_POPULATION_2042': 'sum',
    'CURRENT_DESIGN_FLOW': 'first',
    **{col: 'first' for col in facilities_2022.columns if col not in ['CWNS_ID', 'PERMIT_NUMBER_cwns_clean', 
                                                                      'TOTAL_RES_POPULATION_2022', 'TOTAL_RES_POPULATION_2042',
                                                                      'CURRENT_DESIGN_FLOW']}
})

# Add DUMMY_ID column initialized to CWNS_ID
facilities_2022['DUMMY_ID'] = copy.deepcopy([str(id) for id in facilities_2022['CWNS_ID']])
print(f"{len(facilities_2022)} CWNS facilities with {len(facilities_2022['DUMMY_ID'].unique())} CWNS_IDs after merging with pop served and cleaning")

discharges = pd.read_csv('data/2022CWNS_NATIONAL_APR2024/DISCHARGES.csv', encoding='latin1', low_memory=False)
discharges = discharges[discharges['STATE_CODE']=='CA']
discharges['DISCHARGES_TO_CWNSID'] = pd.to_numeric(discharges['DISCHARGES_TO_CWNSID'], errors='coerce').astype('Int64')
discharges['CWNS_ID'] = pd.to_numeric(discharges['CWNS_ID'], errors='coerce').astype('Int64')
discharges['DUMMY_ID'] = copy.deepcopy([str(id) for id in discharges['CWNS_ID']])
discharges['DISCHARGES_TO_DUMMY_ID'] = copy.deepcopy([str(id) for id in discharges['DISCHARGES_TO_CWNSID']])

all_new_facility_rows = []

# First handle facilities with multiple types
facility_types_order = {
    # Brown collection types
    'Collection: Separate Sewers': 1,
    'Collection: Pump Stations': 1, 
    'Collection: Combined Sewers': 1,
    'Collection: Interceptor Sewers': 1,
    
    # Orange OWTS types
    'Onsite Wastewater Treatment System': 2,
    'Phase II MS4': 2,
    'Phase I MS4': 2,
    'Non-traditional MS4': 2,
    'Sanitary Landfills': 2,
    'Honey Bucket Lagoon': 2,
    
    # Blue treatment types
    'Treatment Plant': 3,
    'Biosolids Handling Facility': 3,
    'Clustered System': 3,
    
    # Grey storage types
    'Storage Tanks': 4,
    'Storage Facility': 4,
    
    # Purple reuse types
    'Water Reuse': 5,
    'Resource Extraction': 5,
    'Desalination - WW': 5,
    
    # Black other types
    'Other': 6
}

print(str(len(discharges)) + ' discharges before adding dummies')
# Create mappings of original CWNS_ID to treatment plant and reuse facility IDs
treatment_plant_mapping = {}
reuse_mapping = {}

# Handle facilities with multiple types first
facilities_processed = 0
for cwns_id, group in facilities_2022.groupby('CWNS_ID'):
    facilities_processed += 1
    if facilities_processed % 5000 == 0:
        print(str(facilities_processed) + ' facilities_processed')
    if len(group) > 1:  # if there is more than one facility type
        # Sort facility types by priority, excluding pump stations which will be handled separately
        sorted_types = sorted(group['FACILITY_TYPE'][group['FACILITY_TYPE'] != 'Collection: Pump Stations'].unique(), 
                            key=lambda x: facility_types_order.get(x, 999))
        
        prev_cwns = cwns_id
        primary_dummy_id = None
        
        # Keep track of types we've already processed
        processed_types = {}
        
        # First create nodes for each facility type
        for t, fac_type in enumerate(sorted_types):
            new_dummy_id = str(cwns_id) + 't' + str(t)
            processed_types[fac_type] = new_dummy_id
            
            # Update DUMMY_ID and name for this facility type
            mask = (facilities_2022['CWNS_ID'] == cwns_id) & (facilities_2022['FACILITY_TYPE'] == fac_type)
            facilities_2022.loc[mask, 'DUMMY_ID'] = new_dummy_id
            # Only update name if it doesn't already contain the facility type
            mask_name = mask & ~facilities_2022['FACILITY_NAME'].str.contains(f'({fac_type})', regex=False, na=False)
            facilities_2022.loc[mask_name, 'FACILITY_NAME'] = facilities_2022.loc[mask_name, 'FACILITY_NAME'] + ' (' + fac_type +')'
            
            # Track treatment plant ID if found
            if fac_type == 'Treatment Plant':
                treatment_plant_mapping[cwns_id] = new_dummy_id
            # Track reuse facility ID if found
            elif 'reuse' in fac_type.lower() or 'reclaim' in fac_type.lower():
                reuse_mapping[cwns_id] = new_dummy_id
            
            # Update primary_dummy_id to latest
            primary_dummy_id = new_dummy_id
        
        # Then create connections between consecutive types
        for t in range(len(sorted_types)-1):
            fac_type1 = sorted_types[t]
            fac_type2 = sorted_types[t+1]
            
            # Add connection between the two facility types
            new_discharge = pd.DataFrame({
                'CWNS_ID': [cwns_id],
                'DUMMY_ID': [processed_types[fac_type1]],
                'DISCHARGES_TO_CWNSID': [cwns_id],
                'DISCHARGES_TO_DUMMY_ID': [processed_types[fac_type2]],
                'DISCHARGE_TYPE': [f'Internal connection from {fac_type1} to {fac_type2}'],
                'PRESENT_DISCHARGE_PERCENTAGE': [100]
            })
            discharges = pd.concat([discharges, new_discharge], ignore_index=True)

# Update external connections to use the mapped dummy IDs
for cwns_id in treatment_plant_mapping:
    # For source connections
    mask = (discharges['CWNS_ID'] == cwns_id) & (discharges['DISCHARGES_TO_CWNSID'] != cwns_id)
    # Check if destination is a reuse facility
    reuse_mask = mask & discharges['DISCHARGE_TYPE'].str.contains('reuse|reclaim|recycle', case=False, na=False)
    # Use reuse mapping for reuse destinations if available
    if cwns_id in reuse_mapping:
        discharges.loc[reuse_mask, 'DUMMY_ID'] = reuse_mapping[cwns_id]
    # Use treatment plant mapping for all other external connections
    discharges.loc[mask & ~reuse_mask, 'DUMMY_ID'] = treatment_plant_mapping[cwns_id]
    
    # For destination connections
    mask = (discharges['DISCHARGES_TO_CWNSID'] == cwns_id) & (discharges['CWNS_ID'] != cwns_id)
    # Use treatment plant mapping for all incoming connections
    discharges.loc[mask, 'DISCHARGES_TO_DUMMY_ID'] = treatment_plant_mapping[cwns_id]

# beyond this point, should only be using DUMMY_ID for constructing map
print(str(len(discharges)) + ' discharges after adding multiple facility types')
print(f"{len(facilities_2022)} CWNS facilities with {len(facilities_2022['DUMMY_ID'].unique())} CWNS_IDs after merging with pop served and cleaning")

# add final discharges
# loop through FACILITIES
for _, facility in facilities_2022.iterrows():
    if pd.isna(facility['DUMMY_ID']):
        continue
        
    facility_discharges = discharges[discharges['DUMMY_ID'] == facility['DUMMY_ID']]
    facility_final_discharges = facility_discharges[facility_discharges['DISCHARGES_TO_CWNSID'].isna()]
    
    # loop through DISCHARGES for that facility
    d_count = 0
    for d, discharge in facility_final_discharges.iterrows():
        d_count += 1

        new_DUMMY_ID = facility['DUMMY_ID'] + 'd' + str(d_count) # Add discharge suffix to the facility's dummy ID
        
        # Create new facility row for the discharge endpoint
        new_facility_row = facility.copy()
        new_facility_row['DUMMY_ID'] = new_DUMMY_ID
        new_facility_row['FACILITY_NAME'] = discharge['DISCHARGE_TYPE']
        new_facility_row['FACILITY_TYPE'] = 'Reuse' if 'Reuse' in discharge['DISCHARGE_TYPE'] else 'Ocean Discharge' if 'Ocean' in discharge['DISCHARGE_TYPE'] else 'Other'
        new_facility_row['PERMIT_NUMBER'] = None
        new_facility_row['CURRENT_DESIGN_FLOW'] = None
        all_new_facility_rows.append(new_facility_row)
        
        # Update the discharge's DISCHARGES_TO_DUMMY_ID
        discharges.loc[d, 'DISCHARGES_TO_DUMMY_ID'] = new_DUMMY_ID

# Merge facilities into discharges
discharges = discharges.merge(facilities_2022[['DUMMY_ID', 'COUNTY_NAME']].drop_duplicates(), on='DUMMY_ID', how='left')

if all_new_facility_rows:
    all_new_facility_rows_df = pd.DataFrame(all_new_facility_rows)
    facilities_2022 = pd.concat([facilities_2022, all_new_facility_rows_df])
print(str(len(facilities_2022))+' facilities after adding dummy rows for discharges')

# save facility list to CSV
facilities_2022[['CWNS_ID', 'DUMMY_ID', 'FACILITY_NAME','PERMIT_NUMBER','TOTAL_RES_POPULATION_2022','FACILITY_TYPE', 'CURRENT_DESIGN_FLOW', 'COUNTY_NAME','STATE_CODE']].to_csv('processed_data/facilities_2022_merged.csv', index=False)

print('building sewershed map')
# BUILD SEWERSHED MAP

def add_connection(row):
    connection = [row['DUMMY_ID'], row['DISCHARGES_TO_DUMMY_ID'], row['PRESENT_DISCHARGE_PERCENTAGE']]
    return connection
sewershed_map = {}
nodes_already_mapped = []

# loop through all discharge rows, to add a connection for each
for _, row in discharges.iterrows():
    discharge_from_id = row["DUMMY_ID"]
    discharges_to = row["DISCHARGES_TO_DUMMY_ID"]

    # Skip if either ID is NA
    if pd.isna(discharge_from_id) or pd.isna(discharges_to):
        continue

    if (discharge_from_id not in nodes_already_mapped and discharges_to not in nodes_already_mapped): 
        # create a new sewershed if the current node isn't in list of previously mapped nodes
        new_sewershed_id = len(sewershed_map) + 1
        sewershed_map[new_sewershed_id] = {
            "nodes": set([discharge_from_id, discharges_to]),
            "connections": [add_connection(row)],
        }
        nodes_already_mapped.extend([discharge_from_id, discharges_to])
    else: # add to existing sewershed
        for sewershed_info in sewershed_map.values():
            if (discharge_from_id in sewershed_info["nodes"] or discharges_to in sewershed_info["nodes"]):
                sewershed_info["nodes"].update([discharge_from_id, discharges_to])
                sewershed_info["connections"].append(add_connection(row))
                nodes_already_mapped.extend([discharge_from_id, discharges_to])
                break

# Consolidate sewersheds with redundant nodes
print(f'{len(sewershed_map)} sewersheds before combining combining sewersheds w/ repetitive nodes')
DUMMY_IDS = list(sewershed_map.keys())
for i in range(len(DUMMY_IDS)):
    for j in range(i + 1, len(DUMMY_IDS)):
        id1, id2 = DUMMY_IDS[i], DUMMY_IDS[j]
        if id1 in sewershed_map and id2 in sewershed_map:
            if sewershed_map[id1]["nodes"] & sewershed_map[id2]["nodes"]:
                # Merge sewersheds
                sewershed_map[id1]["nodes"].update(sewershed_map[id2]["nodes"])
                sewershed_map[id1]["connections"].extend(sewershed_map[id2]["connections"])
                del sewershed_map[id2]
print(f'{len(sewershed_map)} sewersheds after combining combining sewersheds w/ repetitive nodes')

new_sewershed_map = {}
state_county_used = {}

print('getting state and county info for sewershed nodes')
for _sewershed_info in sewershed_map.values():
    # Get state and county info for nodes in this sewershed
    node_info = []
    for node in _sewershed_info["nodes"]:
        node_data = discharges[discharges['DUMMY_ID'] == node][['STATE_CODE', 'COUNTY_NAME']]
        if not node_data.empty:
            node_info.append(node_data.iloc[0].to_dict())
    
    node_info_df = pd.DataFrame(node_info)
    
    # Get most common state and county
    if len(node_info_df) > 0:
        state_counts = node_info_df["STATE_CODE"].value_counts()
        primary_state = state_counts.index[0] if len(state_counts) > 0 else "Unspecified"
        
        county_counts = node_info_df[
            node_info_df["STATE_CODE"] == primary_state
        ]["COUNTY_NAME"].value_counts()
        primary_county = county_counts.index[0] if len(county_counts) > 0 else "Unspecified"
    else:
        primary_state = "Unspecified"
        primary_county = "Unspecified"

    # # TODO: check if necessary
    # primary_state = "Unspecified" if pd.isna(primary_state) else primary_state
    # primary_county = "Unspecified" if pd.isna(primary_county) else primary_county
    
    # Create state-county key for counting
    state_county_key = f"{primary_state}_{primary_county}"
    state_county_used[state_county_key] = state_county_used.get(state_county_key, 0) + 1
    
    # new sewershed name with state and county
    new_name = f"{primary_state} - {primary_county} County Sewershed {state_county_used[state_county_key]}"
    
    # Add flow and population data for each node in sewershed map, from facilities list
    node_data = {}
    for node in _sewershed_info["nodes"]:
        node_data[node] = {}
        facility_mask = facilities_2022['DUMMY_ID'] == node
        if not facilities_2022[facility_mask].empty:
            facility = facilities_2022[facility_mask].iloc[0]
            for key in ['CURRENT_DESIGN_FLOW', 'TOTAL_RES_POPULATION_2022', 'PERMIT_NUMBER', 'CWNS_ID', 'DUMMY_ID', 'FACILITY_NAME', 'FACILITY_TYPE']:
                node_data[node][key] = facility[key]
            
            # Determine color based on facility type/name
            facility_type = facility['FACILITY_TYPE']
            facility_name = facility['FACILITY_NAME']
            
            if 'Reuse' in facility_type or 'Reuse' in facility_name:
                node_data[node]['color'] = '#9370DB'  # Purple
            elif 'Ocean Discharge' in facility_type or 'Ocean Discharge' in facility_name:
                node_data[node]['color'] = '#90EE90'  # Green
            elif 'Treatment Plant' in facility_type or '(Treatment Plant)' in facility_name:
                node_data[node]['color'] = '#ADD8E6'  # Blue
            elif any(x in facility_type or x in facility_name for x in ['Collection:', 'Pump Station']):
                node_data[node]['color'] = '#C4A484'  # Brown
            elif any(x in facility_type or x in facility_name for x in ['Storage']):
                node_data[node]['color'] = '#808080'  # Grey
            elif any(x in facility_type or x in facility_name for x in ['MS4', 'Onsite', 'Landfill', 'Honey Bucket']):
                node_data[node]['color'] = '#FFD580'  # Orange
            elif 'Outfall' in facility_type or 'Outfall' in facility_name:
                node_data[node]['color'] = '#90EE90'  # Green
            else:
                node_data[node]['color'] = '#FFFFC5'  # Light yellow for other/unknown
        else:
            for key in ['CURRENT_DESIGN_FLOW', 'TOTAL_RES_POPULATION_2022', 'PERMIT_NUMBER', 'CWNS_ID', 'DUMMY_ID', 'FACILITY_NAME', 'FACILITY_TYPE']:
                node_data[node][key] = None
            node_data[node]['color'] = '#FFFFC5'  # Light yellow for missing data

    _sewershed_info["node_data"] = node_data
    new_sewershed_map[new_name] = _sewershed_info

sewershed_map = new_sewershed_map


print('saving sewershed map')
pickle.dump(sewershed_map, open('processed_data/sewershed_map.pkl', 'wb'))