# US Sewersheds

Process and analyze US sewershed data from the Clean Watersheds Needs Survey (CWNS).

## Overview

This repository includes code to visualize sewershed interconnections in the US based on the 2022 Clean Watershed Needs Survey. The us_sewersheds folder includes two scripts:

1. **merge_cwns_data.py**
   - Merges multiple sources for population served into the primary facilities list.
   - Functions:
     - `main(state=None)`: Main processing function that can process all states or a single state.
     - `merge_population_data(facilities_df, ww_df, sso_df)`: Merges population data from multiple sources.
     - `build_sewershed_map(facilities_df)`: Creates network connections between treatment facilities.
   - Required inputs:
     - data/2022CWNS_NATIONAL_APR2024: Clean Watersheds Needs Survey 2022 dataset
       - FACILITIES.csv: Main facilities data
       - FACILITY_PERMIT.csv: Facility permit information
       - AREAS_COUNTY.csv: County area information
       - FACILITY_TYPES.csv: Facility type information
       - FLOW.csv: Flow data
       - POPULATION_WASTEWATER.csv: Wastewater population data
       - POPULATION_WASTEWATER_CONFIRMED.csv: Confirmed wastewater population data
       - POPULATION_DECENTRALIZED.csv: Decentralized population data
       - DISCHARGES.csv: Discharge information
2. **sewersheds_app.py**
   - Deploys Streamlit application to visualize different sewersheds in the US, by state and county

## Installation