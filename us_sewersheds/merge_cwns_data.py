import pandas as pd
import numpy as np
import pickle

# CIWQS
# ciwqs_facilities = pd.read_csv('data/facilities_list/NPDES+WDR Facilities List_20240906.csv')
# print(f'Length of full list: {len(ciwqs_facilities)}')
# print(f' Unique FACILITY ID: {len(ciwqs_facilities['FACILITY ID'].unique())}')
# print(f' Unique WDID: {len(ciwqs_facilities['WDID'].unique())}')
# print(f' Unique ORDER #: {len(ciwqs_facilities['ORDER #'].unique())}')
# ciwqs_facilities.drop_duplicates(subset='WDID', keep='first', inplace=True) # drop duplicate WDIDs

#CWNS 2022
facilities = pd.read_csv('data/cwns/CA_2022CWNS_APR2024/FACILITIES.csv', encoding='latin1', low_memory=False)[['CWNS_ID', 'FACILITY_NAME']]
facility_permits = pd.read_csv('data/cwns/CA_2022CWNS_APR2024/FACILITY_PERMIT.csv', encoding='latin1', low_memory=False).drop(columns=['FACILITY_ID', 'STATE_CODE'])
areas_county = pd.read_csv('data/cwns/CA_2022CWNS_APR2024/AREAS_COUNTY.csv', encoding='latin1', low_memory=False)
facility_types = pd.read_csv('data/cwns/CA_2022CWNS_APR2024/FACILITY_TYPES.csv', encoding='latin1', low_memory=False)
discharges = pd.read_csv('data/cwns/CA_2022CWNS_APR2024/DISCHARGES.csv', encoding='latin1', low_memory=False)
discharges['DISCHARGES_TO_CWNSID'] = pd.to_numeric(discharges['DISCHARGES_TO_CWNSID'], errors='coerce').astype('Int64')
discharges['CWNS_ID'] = pd.to_numeric(discharges['CWNS_ID'], errors='coerce').astype('Int64')
pop_served_cwns = pd.concat([
    pd.read_csv('data/cwns/CA_2022CWNS_APR2024/POPULATION_WASTEWATER.csv', encoding='latin1', low_memory=False),
    pd.read_csv('data/cwns/CA_2022CWNS_APR2024/POPULATION_WASTEWATER_CONFIRMED.csv', encoding='latin1', low_memory=False),
    pd.read_csv('data/cwns/CA_2022CWNS_APR2024/POPULATION_DECENTRALIZED.csv', encoding='latin1', low_memory=False)
])

#CWNS 2012
cwns_2012 = pd.read_csv('data/cwns/2012/Facility_Details_CA_population.csv')

## WW Surveillance
# pop_served_ww_surveillance = pd.read_csv('data/ww_surveillance/wastewatersurveillancecalifornia.csv', low_memory=False)[['epaid', 'population_served']].dropna().drop_duplicates(subset='epaid')

# # SSO Questionnaire
# questionnaire = pd.read_csv('data/sso/Questionnaire.txt', low_memory=False, delimiter='\t')[['Wdid','SSOq Population Served']]

# ## AVR
# avr_influent = pd.read_csv('data/avr/avr_export_influent.csv', encoding='latin-1')
# avr_facility = pd.read_csv('data/avr/avr_export_facility.csv', encoding='latin-1')
# # create mapping from FACILITY PLACE ID to GLOBAL ID
# facility_mapping = avr_facility[['FACILITY PLACE ID', 'GLOBAL ID']].drop_duplicates(subset='FACILITY PLACE ID')
# # map the GLOBAL ID from avr_influent to the FACILITY PLACE ID in avr_facility
# avr_influent['FACILITY PLACE ID'] = avr_influent['GLOBAL ID'].map(facility_mapping.set_index('GLOBAL ID')['FACILITY PLACE ID'])

# # Drop rows where 'FACILITY PLACE ID' is null after mapping
# avr_influent = avr_influent.dropna(subset=['FACILITY PLACE ID'])

# # Ensure 'REPORTING YEAR' is numeric
# avr_influent['REPORTING YEAR'] = pd.to_numeric(avr_influent['REPORTING YEAR'], errors='coerce')

# facility_max_years = avr_influent.groupby('FACILITY PLACE ID')['REPORTING YEAR'].max().reset_index()

# avr_influent_recent = pd.merge(avr_influent, facility_max_years, 
#                                on=['FACILITY PLACE ID', 'REPORTING YEAR'], 
#                                how='inner',  # Use inner join to keep only matching rows
#                                suffixes=('', '_max'))
# avr_influent_july_recent = avr_influent_recent[avr_influent_recent['REPORTING MONTH'] == 7]

# IMPORT DATA
facility_permit = pd.read_csv('data/cwns/CA_2022CWNS_APR2024/FACILITY_PERMIT.csv', encoding='latin1', low_memory=False)
facilities = pd.read_csv('data/cwns/CA_2022CWNS_APR2024/FACILITIES.csv', encoding='latin1', low_memory=False)
areas_county = pd.read_csv('data/cwns/CA_2022CWNS_APR2024/AREAS_COUNTY.csv', encoding='latin1', low_memory=False)
facility_types = pd.read_csv('data/cwns/CA_2022CWNS_APR2024/FACILITY_TYPES.csv', encoding='latin1', low_memory=False)
discharges = pd.read_csv('data/cwns/CA_2022CWNS_APR2024/DISCHARGES.csv', encoding='latin1', low_memory=False)
population_wastewater = pd.read_csv('data/cwns/CA_2022CWNS_APR2024/POPULATION_WASTEWATER.csv', encoding='latin1', low_memory=False)
discharges['DISCHARGES_TO_CWNSID'] = pd.to_numeric(discharges['DISCHARGES_TO_CWNSID'], errors='coerce').astype('Int64')
discharges['CWNS_ID'] = pd.to_numeric(discharges['CWNS_ID'], errors='coerce').astype('Int64')
discharges = discharges.merge(facilities, on='CWNS_ID', how='left')
discharges = discharges.merge(areas_county, on='CWNS_ID', how='left')
# save intermediate file for plot_sewersheds.py
facilities[['CWNS_ID', 'FACILITY_NAME']].to_csv('processed_data/cwns_facilities_merged.csv', index=False)

# clean CWNS population data
print(f'{len(facilities)} CWNS facilities')
facilities = facilities.merge(facility_permits, on='CWNS_ID', how='left')
print(f'{len(facilities)} CWNS facilities after merging with facility_permits')
facilities = facilities.merge(facility_types, on='CWNS_ID', how='left')
print(f'{len(facilities)} CWNS facilities after merging with facility_types')
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
    **{col: 'first' for col in cwns_facilities.columns if col not in ['CWNS_ID', 'PERMIT_NUMBER_cwns_clean', 
                                                                      'TOTAL_RES_POPULATION_2022', 'TOTAL_RES_POPULATION_2042']}
})
print(f'{len(cwns_facilities)} CWNS facilities after merging with pop served and cleaning')
# print(f'{len(ciwqs_facilities)} CIWQS WDIDs')

# MERGE ON CWNS ID for 2012 data
cwns_2012_mapping = cwns_2012[['CWNS Number', 'Present Residential Total Receiving Treatment Population']].drop_duplicates(subset='CWNS Number')
cwns_2012_mapping.columns = ['CWNS_ID', 'TOTAL_RES_POPULATION_2012']
cwns_facilities = cwns_facilities.merge(cwns_2012_mapping, on='CWNS_ID', how='left')

# # Merge on 'NPDES # CA#' and 'ORDER #' for 2022 data
# ciwqs_facilities = ciwqs_facilities.merge(cwns_facilities[['PERMIT_NUMBER_cwns_clean', 'TOTAL_RES_POPULATION_2022', 'TOTAL_RES_POPULATION_2012', 'TOTAL_RES_POPULATION_2042']], 
#                                           left_on='NPDES # CA#', right_on='PERMIT_NUMBER_cwns_clean', how='left')
# # Add matches for 'ORDER #'
# na_mask = ciwqs_facilities['TOTAL_RES_POPULATION_2022'].isna()
# order_match_dict = cwns_facilities.set_index('PERMIT_NUMBER_cwns_clean')[['TOTAL_RES_POPULATION_2022', 'TOTAL_RES_POPULATION_2012', 'TOTAL_RES_POPULATION_2042']].to_dict()
# ciwqs_facilities.loc[na_mask, 'TOTAL_RES_POPULATION_2022'] = ciwqs_facilities.loc[na_mask, 'ORDER #'].map(order_match_dict['TOTAL_RES_POPULATION_2022'])
# ciwqs_facilities.loc[na_mask, 'TOTAL_RES_POPULATION_2012'] = ciwqs_facilities.loc[na_mask, 'ORDER #'].map(order_match_dict['TOTAL_RES_POPULATION_2012'])
# ciwqs_facilities.loc[na_mask, 'TOTAL_RES_POPULATION_2042'] = ciwqs_facilities.loc[na_mask, 'ORDER #'].map(order_match_dict['TOTAL_RES_POPULATION_2042'])
# print(f'{len(ciwqs_facilities)} CIWQS WDIDs after merge with cwns_facilities')

# # WW Surveillance
# pop_served_ww_surveillance.rename(columns={'epaid': 'PERMIT_NUMBER_ww_surveillance'}, inplace=True)
# ciwqs_facilities = ciwqs_facilities.merge(pop_served_ww_surveillance, left_on='NPDES # CA#', right_on='PERMIT_NUMBER_ww_surveillance', how='left')
# print(f'{len(ciwqs_facilities)} CIWQS WDIDs after merging with pop_served_ww_surveillance')

# # SSO Questionnaire - right now there are no matches as the facilities list doesn't include SSO, but this is a placeholder
# ciwqs_facilities['WDID'] = ciwqs_facilities['WDID'].astype(str)
# questionnaire['Wdid'] = questionnaire['Wdid'].astype(str)
# ciwqs_facilities = ciwqs_facilities.merge(questionnaire, left_on='WDID', right_on='Wdid', how='left')

# # AVR
# ciwqs_facilities['AVR_JULY_FLOW'] = ciwqs_facilities['FACILITY ID'].map(
#     avr_influent_july_recent.set_index('FACILITY PLACE ID')['INFLUENT VOLUME']
# )

# # Create a unified population data column, first using CWNS where available, then SSO Questionnaire, then WW Surveillance
# population_source_prioritization = {'CWNS': 'TOTAL_RES_POPULATION_2022', 'SSO Questionnaire': 'SSOq Population Served', 'Wastewater Surveillance': 'population_served'}
# ciwqs_facilities[['POPULATION_SERVED', 'POPULATION_SOURCE']] = np.nan, 'None'
# for source, column in population_source_prioritization.items():
#     mask = ciwqs_facilities['POPULATION_SERVED'].isna()
#     ciwqs_facilities.loc[mask, 'POPULATION_SERVED'] = pd.to_numeric(ciwqs_facilities.loc[mask, column], errors='coerce')
#     ciwqs_facilities.loc[mask & ciwqs_facilities['POPULATION_SERVED'].notna(), 'POPULATION_SOURCE'] = source
# ciwqs_facilities['POPULATION_SERVED'] = ciwqs_facilities['POPULATION_SERVED'].astype('float64')


# # sum up the population column in ciwqs_facilities
# print(f'Total Population Served for facilities with non-nan population served: {int(ciwqs_facilities[ciwqs_facilities['POPULATION_SERVED'].notna()]['POPULATION_SERVED'].sum()):,}')

# # sum up the total design flow column in ciwqs_facilities where popultion served is not nan
# print(f'Total Design Flow for all facilities: {ciwqs_facilities['DESIGN FLOW'].sum():,.0f} MGD')
# print(f'Total Design Flow for facilities with non-nan population served: {ciwqs_facilities[ciwqs_facilities['POPULATION_SERVED'].notna()]['DESIGN FLOW'].sum():,.0f} MGD')
# print(f'Percentage of total design flow where population served is not nan: {ciwqs_facilities[ciwqs_facilities['POPULATION_SERVED'].notna()]['DESIGN FLOW'].sum() / ciwqs_facilities['DESIGN FLOW'].sum() * 100:.2f}%')

# # save facilities with population served
# ciwqs_facilities.to_csv('processed_data/facilities_list_with_population_served.csv', index=False)

# # for facilities with no population served, estimate the population using "DESIGN FLOW" (MGD) / 55 gallons per person per day
# # https://www.waterboards.ca.gov/conservation/regs/docs/appendix-3-013022.pdf

# gppd = 55 # gallons per person per day
# ciwqs_facilities['POPULATION_SERVED_ESTIMATE'] = (ciwqs_facilities['DESIGN FLOW'] * 1000000 / gppd).fillna(0).astype(int)

# #  sum up the population served estimate for a gut check
# print(f'Total Population Estimate for facilities with no population served: {int(ciwqs_facilities[ciwqs_facilities['POPULATION_SERVED'].isna()]['POPULATION_SERVED_ESTIMATE'].sum()):,}')
# print(f'Total Population Estimate for all facilities: {int(ciwqs_facilities['POPULATION_SERVED_ESTIMATE'].sum()):,}')

# # add to the total population served
# ciwqs_facilities['POPULATION_SERVED'] = ciwqs_facilities['POPULATION_SERVED'].fillna(ciwqs_facilities['POPULATION_SERVED_ESTIMATE'])

# # drop the estimate column
# ciwqs_facilities.drop(columns=['POPULATION_SERVED_ESTIMATE'], inplace=True)

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

# save sewershed map to pickle file
pickle.dump(sewershed_map, open('processed_data/sewershed_map.pkl', 'wb'))

# # Calculate percentage of NPDES and WDR facilities with non-zero design flow
# total_wdr = len(ciwqs_facilities[ciwqs_facilities["PROGRAM"].str.contains("WDR", na=False)])
# total_npdes = len(ciwqs_facilities[ciwqs_facilities["PROGRAM"].str.contains("NPD", na=False)])

# wdr_with_flow = len(ciwqs_facilities[(ciwqs_facilities["DESIGN FLOW"].notna() & (ciwqs_facilities["DESIGN FLOW"] > 0) & ciwqs_facilities["PROGRAM"].str.contains("WDR", na=False))])
# npdes_with_flow = len(ciwqs_facilities[(ciwqs_facilities["DESIGN FLOW"].notna() & (ciwqs_facilities["DESIGN FLOW"] > 0) & ciwqs_facilities["PROGRAM"].str.contains("NPD", na=False))])

# wdr_percentage = (wdr_with_flow / total_wdr) * 100 if total_wdr > 0 else 0
# npdes_percentage = (npdes_with_flow / total_npdes) * 100 if total_npdes > 0 else 0

# print(f'{wdr_percentage:.2f}% of WDR facilities have a non-zero design flow ({wdr_with_flow} out of {total_wdr})')
# print(f'{npdes_percentage:.2f}% of NPDES facilities have a non-zero design flow ({npdes_with_flow} out of {total_npdes})')