# US Sewersheds

Process and analyze US sewershed data from the Clean Watersheds Needs Survey (CWNS).

## Overview

This repository includes code to visualize sewershed interconnections in the US based on the 2022 Clean Watershed Needs Survey. The us_sewersheds folder includes two scripts:

### Data Processing Logic

The system handles facility relationships using a **nested facility types structure**:

1. **Multi-type Facilities**: Some facilities have multiple types (e.g., Treatment Plant + Collection + Reuse). Instead of creating separate dummy IDs, each facility now contains a nested `TYPES` array where each type is a separate node with its own coordinates and properties:
   - **Processing Order**: Collection Systems (order 0) → Interceptor Systems (order 1) → Treatment Plants (order 3) → Water Reuse (order 5) → Outfalls (order 6)
   - **Coordinate Spacing**: Each facility type gets slightly offset coordinates for visual separation
   - **Node IDs**: Use format `CWNS_ID_type_X` where X is the index in the TYPES array

2. **Final Discharges**: Creates additional facility type nodes for discharges to the environment (not to other facilities). Each unique discharge type gets a separate node with facility type determined by discharge type keywords (e.g., "Reuse", "Ocean Discharge").

3. **Connection Logic**: 
   - **External Connections**: Between different CWNS_IDs using connection rules to select appropriate source/target facility types
   - **Internal Connections**: Between facility types within the same CWNS_ID based on processing order
   - **Mass Balance**: Internal flow percentages calculated as 100% minus external flow to maintain mass balance

4. **Deduplication**: Uses set-based operations throughout to automatically prevent duplicate connections and nodes.

1. **merge_cwns_data.py**
   - Merges multiple sources for population served into the primary facilities list.
   - Functions:
     - `main(state=None)`: Main processing function that can process all states or a single state.
     - `process_multi_type_facilities(data)`: Creates nested facility types structure for multi-type facilities.
     - `create_sewershed_map(data, processed_facilities)`: Creates network connections between facilities.
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
   - Supports both Network Graph (hierarchical) and Network Map (geographic) views

## Installation

### From PyPI
```bash
pip install us-sewersheds
```

### From Source