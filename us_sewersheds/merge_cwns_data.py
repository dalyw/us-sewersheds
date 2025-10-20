import pandas as pd
from collections import Counter
import pickle

from plotting_configs import (
    get_node_color,
    DEFAULT_NODE_COLOR,
    get_facility_type_order,
)


file_configs = {
    "facilities": {
        "file": "FACILITIES.csv",
        "columns": ["CWNS_ID", "FACILITY_NAME", "STATE_CODE"],
    },
    "permits": {
        "file": "FACILITY_PERMIT.csv",
        "columns": ["CWNS_ID", "PERMIT_NUMBER"],
        "groupby": {"CWNS_ID": ["PERMIT_NUMBER"]},
    },
    "counties": {
        "file": "AREAS_COUNTY.csv",
        "columns": ["CWNS_ID", "COUNTY_NAME", "COUNTY_PRIMARY_FLAG"],
        "filter": {"COUNTY_PRIMARY_FLAG": "Y"},
        "select": ["CWNS_ID", "COUNTY_NAME"],
    },
    "types": {
        "file": "FACILITY_TYPES.csv",
        "columns": ["CWNS_ID", "FACILITY_TYPE"],
    },
    "flow": {
        "file": "FLOW.csv",
        "columns": ["CWNS_ID", "FLOW_TYPE", "CURRENT_DESIGN_FLOW"],
        "filter": {"FLOW_TYPE": "Total Flow"},
        "select": ["CWNS_ID", "CURRENT_DESIGN_FLOW"],
    },
    "physical_location": {
        "file": "PHYSICAL_LOCATION.csv",
        "columns": [
            "CWNS_ID",
            "LATITUDE",
            "LONGITUDE",
            "CITY",
            "STATE_CODE",
        ],
    },
    "population_wastewater": {
        "file": "POPULATION_WASTEWATER.csv",
        "columns": [
            "CWNS_ID",
            "TOTAL_RES_POPULATION_2022",
            "TOTAL_RES_POPULATION_2042",
        ],
    },
    "population_wastewater_confirmed": {
        "file": "POPULATION_WASTEWATER_CONFIRMED.csv",
        "columns": [
            "CWNS_ID",
            "TOTAL_RES_POPULATION_2022",
            "TOTAL_RES_POPULATION_2042",
        ],
    },
    "population_decentralized": {
        "file": "POPULATION_DECENTRALIZED.csv",
        "columns": [
            "CWNS_ID",
            "RESIDENTIAL_POP_2022",
            "RESIDENTIAL_POP_2042",
        ],
        "rename": {
            "RESIDENTIAL_POP_2022": "TOTAL_RES_POPULATION_2022",
            "RESIDENTIAL_POP_2042": "TOTAL_RES_POPULATION_2042",
        },
        "select": [
            "CWNS_ID",
            "TOTAL_RES_POPULATION_2022",
            "TOTAL_RES_POPULATION_2042",
        ],
    },
}


def load_cwns_data(data_dir="data/2022CWNS_NATIONAL_APR2024/"):
    """
    Load all CWNS data files from specified directory

    Args:
        data_dir: Directory containing CWNS data files

    Returns:
        Dictionary of DataFrames containing cleaned CWNS data
    """

    # Load all CSV files
    data = {}
    for name, config in file_configs.items():
        try:
            df = pd.read_csv(
                f"{data_dir}{config['file']}",
                encoding="latin1",
                low_memory=False,
            )
            df = df[config["columns"]]

            # Apply filters
            if "filter" in config:
                for col, value in config["filter"].items():
                    df = df[df[col] == value]

            # Drop columns
            if "drop" in config:
                df = df.drop(config["drop"], axis=1)

            # Rename columns
            if "rename" in config:
                df = df.rename(columns=config["rename"])

            # Select final columns
            if "select" in config:
                df = df[config["select"]]

            # Groupby operations
            if "groupby" in config:
                for group_col, agg_cols in config["groupby"].items():
                    # For permits, collect all permit numbers for each facility
                    df = (
                        df.groupby(group_col)[agg_cols[0]]
                        .apply(list)
                        .reset_index()
                    )
                    # Keep lists as they represent multiple permits per facility
            else:
                df = df.drop_duplicates()

            data[name] = df

        except FileNotFoundError:
            raise

    data["facilities"] = normalize_cwns_ids(data["facilities"], ["CWNS_ID"])

    # Consolidate population data files
    pop_dfs = [
        data["population_wastewater"],
        data["population_wastewater_confirmed"],
        data["population_decentralized"],
    ]
    data["population"] = pd.concat(pop_dfs).drop_duplicates(
        subset="CWNS_ID", keep="first"
    )

    # Remove individual population files from data dict
    for pop_key in [
        "population_wastewater",
        "population_wastewater_confirmed",
        "population_decentralized",
    ]:
        del data[pop_key]

    return data


def get_columns_from_configs(
    include_facility_base=True, include_calculated=True
):
    """Generate columns dynamically from file_configs

    Args:
        include_facility_base: Include base facility columns like DUMMY_ID, FACILITY_NAME
        include_calculated: Include calculated columns like POPULATION_PERCENT_INCREASE
    """
    # Aggregate all columns from all configs
    all_columns = set()
    for config_name, config in file_configs.items():
        # Get the final columns after processing (select takes precedence over columns)
        if "select" in config:
            columns = config["select"]
        else:
            columns = config["columns"]
        all_columns.update(columns)

    # Start with base columns if requested
    if include_facility_base:
        result_columns = ["CWNS_ID", "DUMMY_ID", "FACILITY_NAME"]
    else:
        result_columns = []

    # Add all other columns
    for col in sorted(all_columns):
        if col not in result_columns:
            result_columns.append(col)

    # Add calculated columns if requested
    if include_calculated:
        result_columns.append("POPULATION_PERCENT_INCREASE")

    return result_columns


def normalize_cwns_ids(df, cwns_id_columns):
    """Remove leading zeros from CWNS_ID columns to ensure consistent integer representation"""
    df = df.copy()
    for col in cwns_id_columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.replace(
                    ".0", "", regex=False
                )  # Remove .0 from float representations
                .str.lstrip("0")
                .replace("", "0")  # Handle case where all zeros were stripped
                .astype("int64")
            )
    return df


def clean_permit_numbers(facilities_df):
    """Clean permit numbers by removing common patterns and standardizing format"""
    # Simplified patterns using case-insensitive matching
    patterns_to_remove = [
        r"wdr\s*",  # WDR with optional whitespace (covers "WDR ", "WDR-", "WDR")
        r"order\s+wq\s*",  # Order WQ with optional whitespace
        r"wdr\s+order\s+no\.\s*",  # WDR Order No. with optional whitespace
        r"order\s+no\.\s*",  # Order No. with optional whitespace
        r"order\s+",  # Order with required whitespace
        r"no\.\s*",  # No. with optional whitespace (covers "NO. ", "NO.")
        r"dwq-\s*",  # DWQ- with optional whitespace
        r"no\.·",  # NO.· (special character)
        r"\.\s+",  # Period followed by whitespace
    ]
    replacements = {r"·": "-", r"\?": "-"}

    df = facilities_df.copy()
    df["PERMIT_NUMBER_cwns_clean"] = (
        df["PERMIT_NUMBER"]
        .astype(str)
        .replace("(?i)" + "|".join(patterns_to_remove), "", regex=True)
    )
    for old, new in replacements.items():
        df["PERMIT_NUMBER_cwns_clean"] = df[
            "PERMIT_NUMBER_cwns_clean"
        ].str.replace(old, new, regex=True)
    return df


def preprocess_facilities_and_discharges(facilities_df, discharges_df):
    """
    Preprocess facilities and discharges data - separate concerns from sewershed construction

    Args:
        facilities_df: DataFrame containing facility information
        discharges_df: DataFrame containing discharge information

    Returns:
        Tuple containing:
        - Updated facilities DataFrame with all preprocessing complete
        - Updated discharges DataFrame with all preprocessing complete
    """
    facility_types_order = get_facility_type_order()

    # Step 1: Normalize CWNS_IDs to remove leading zeros
    facilities_df = normalize_cwns_ids(facilities_df, ["CWNS_ID"])
    discharges_df = normalize_cwns_ids(
        discharges_df, ["CWNS_ID", "DISCHARGES_TO_CWNSID"]
    )

    # Step 2: Add DUMMY_ID column initialized to CWNS_ID if not exists
    if "DUMMY_ID" not in facilities_df.columns:
        facilities_df["DUMMY_ID"] = facilities_df["CWNS_ID"].astype(str)

    # Step 3: Create unique DUMMY_IDs for each discharge record (per CWNS_ID)
    discharge_counter = {}
    discharge_dummy_ids = []

    for cwns_id in discharges_df["CWNS_ID"]:
        if cwns_id not in discharge_counter:
            discharge_counter[cwns_id] = 0
        discharge_counter[cwns_id] += 1
        discharge_dummy_ids.append(f"{cwns_id}_d{discharge_counter[cwns_id]}")

    discharges_df["DUMMY_ID"] = discharge_dummy_ids
    discharges_df["DISCHARGES_TO_DUMMY_ID"] = discharges_df[
        "DISCHARGES_TO_CWNSID"
    ].astype(str)

    # Step 4: Handle facilities with multiple types
    facilities_df, discharges_df = _process_multi_type_facilities(
        facilities_df, discharges_df, facility_types_order
    )

    # Step 5: Handle final discharges
    facilities_df, discharges_df = _process_final_discharges(
        facilities_df, discharges_df
    )

    # Step 6: Apply node spacing for overlapping coordinates
    facilities_df = _apply_node_spacing(facilities_df)

    return facilities_df, discharges_df


def _process_multi_type_facilities(
    facilities_df, discharges_df, facility_types_order
):
    """Handle facilities with multiple types - create separate nodes for each type"""
    # Pre-compute facility type counts
    facility_type_counts = facilities_df.groupby("CWNS_ID").size()
    multi_type_facilities = facility_type_counts[facility_type_counts > 1].index
    print(
        f"Processing {len(multi_type_facilities)} facilities with multiple types"
    )

    # Collect all changes and apply them at once
    facilities_processed = 0
    new_facilities = []
    new_discharges = []

    for cwns_id in multi_type_facilities:
        facilities_processed += 1
        if facilities_processed % 2000 == 0:
            print(
                f"{facilities_processed} out of {len(multi_type_facilities)} multi-type facilities processed"
            )

        group = facilities_df[facilities_df["CWNS_ID"] == cwns_id]

        if len(group) > 1:
            # Sort facility types by priority, excluding pump stations
            types_in_group = group["FACILITY_TYPE"].unique()
            non_pump_types = [
                t for t in types_in_group if t != "Collection: Pump Stations"
            ]
            sorted_types = sorted(
                non_pump_types, key=lambda x: facility_types_order.get(x, 999)
            )

            if len(sorted_types) <= 1:
                # Keep the original facility if only one type
                original_facility = group.iloc[0].copy()
                original_facility["DUMMY_ID"] = str(
                    cwns_id
                )  # Convert to string for consistency with dummy ids
                new_facilities.append(original_facility.to_frame().T)
                continue

            # Create new facility records for each type
            for t, fac_type in enumerate(sorted_types):
                type_facilities = group[
                    group["FACILITY_TYPE"] == fac_type
                ].copy()
                new_dummy_id = f"{cwns_id}t{t}"
                type_facilities["DUMMY_ID"] = new_dummy_id

                # Update names that don't already contain the facility type
                name_mask = ~type_facilities["FACILITY_NAME"].str.contains(
                    f"({fac_type})", regex=False, na=False
                )
                type_facilities.loc[name_mask, "FACILITY_NAME"] = (
                    type_facilities.loc[name_mask, "FACILITY_NAME"]
                    + f" ({fac_type})"
                )

                new_facilities.append(type_facilities)

            # Create connections between consecutive types
            for t in range(len(sorted_types) - 1):
                fac_type1, fac_type2 = sorted_types[t], sorted_types[t + 1]
                dummy_id1, dummy_id2 = f"{cwns_id}t{t}", f"{cwns_id}t{t+1}"
                new_discharge = pd.DataFrame(
                    {
                        "CWNS_ID": [cwns_id],
                        "DUMMY_ID": [dummy_id1],
                        "DISCHARGES_TO_CWNSID": [cwns_id],
                        "DISCHARGES_TO_DUMMY_ID": [dummy_id2],
                        "DISCHARGE_TYPE": [
                            f"Internal connection from {fac_type1} to {fac_type2}"
                        ],
                        "PRESENT_DISCHARGE_PERCENTAGE": [100],
                    }
                )
                new_discharges.append(new_discharge)

    # Remove original multi-type facilities and add new ones
    print("Applying updates to multi-type facilities")

    # Keep single-type facilities (not in multi_type_facilities)
    single_type_facilities = facilities_df[
        ~facilities_df["CWNS_ID"].isin(multi_type_facilities)
    ].copy()

    # Combine single-type facilities with processed multi-type facilities
    all_facilities = [single_type_facilities] + new_facilities
    facilities_df = pd.concat(all_facilities, ignore_index=True)
    discharges_df = pd.concat(
        [discharges_df] + new_discharges, ignore_index=True
    )

    return facilities_df, discharges_df


def _process_final_discharges(facilities_df, discharges_df):
    """Process final discharges - create new facility nodes for each discharge type"""
    all_new_facility_rows = []

    # Pre-group discharges by DUMMY_ID for efficient lookup
    discharges_by_dummy_id = discharges_df.groupby("DUMMY_ID")

    for dummy_id, facility_group in facilities_df.groupby("DUMMY_ID"):
        if pd.isna(dummy_id):
            continue

        facility = facility_group.iloc[0]

        if dummy_id in discharges_by_dummy_id.groups:
            facility_discharges = discharges_by_dummy_id.get_group(dummy_id)
            facility_final_discharges = facility_discharges[
                facility_discharges["DISCHARGES_TO_CWNSID"].isna()
            ]
        else:
            facility_final_discharges = pd.DataFrame()

        # Process each discharge from treatment facilities
        facility_type = facility.get("FACILITY_TYPE", "").lower()

        # Skip creating dummy IDs for collection systems - they should keep their original CWNS_ID
        if "collection" in facility_type:
            continue

        for d_count, (d, discharge) in enumerate(
            facility_final_discharges.iterrows(), 1
        ):
            new_DUMMY_ID = f"{facility['DUMMY_ID']}d{d_count}"
            discharge_type = discharge["DISCHARGE_TYPE"]

            # Create new facility row
            new_facility_row = facility.copy()
            new_facility_row["DUMMY_ID"] = new_DUMMY_ID
            new_facility_row["FACILITY_NAME"] = (
                f"{facility['FACILITY_NAME']} - {discharge_type}"
            )
            new_facility_row["PERMIT_NUMBER"] = None
            new_facility_row["CURRENT_DESIGN_FLOW"] = None

            # Determine facility type based on discharge type
            discharge_type_lower = discharge_type.lower()
            if "reuse" in discharge_type_lower:
                new_facility_row["FACILITY_TYPE"] = "Reuse"
            elif "outfall" in discharge_type_lower:
                new_facility_row["FACILITY_TYPE"] = (
                    "Ocean Discharge" if "Ocean" in discharge_type else "Other"
                )
            else:
                new_facility_row["FACILITY_TYPE"] = "Other"

            all_new_facility_rows.append(new_facility_row)
            discharges_df.loc[d, "DISCHARGES_TO_DUMMY_ID"] = new_DUMMY_ID

    # Add the new discharge facility rows to the facilities DataFrame
    if all_new_facility_rows:
        new_facilities_df = pd.DataFrame(all_new_facility_rows)
        facilities_df = pd.concat(
            [facilities_df, new_facilities_df], ignore_index=True
        )
        print(f"Added {len(all_new_facility_rows)} new discharge facility rows")

    return facilities_df, discharges_df


def _apply_node_spacing(facilities_df):
    """Apply node spacing for overlapping coordinates"""
    coord_groups = facilities_df.groupby(["LATITUDE", "LONGITUDE"])
    for (lat, lon), group in coord_groups:
        if len(group) > 1:  # Only process groups with multiple facilities
            num_nodes = len(group)
            spacing_distance = 0.01  # Approximately 1km at mid-latitudes
            start_lon = lon - (spacing_distance * (num_nodes - 1) / 2)
            sorted_group = group.sort_values("DUMMY_ID")
            for i, (idx, facility) in enumerate(sorted_group.iterrows()):
                new_lon = start_lon + (i * spacing_distance)
                facilities_df.loc[idx, "LONGITUDE"] = new_lon
                vertical_offset = (i % 2) * 0.0002  # Alternate up/down
                facilities_df.loc[idx, "LATITUDE"] = lat + vertical_offset

    return facilities_df


def add_connection(row):
    return [
        row["DUMMY_ID"],
        row["DISCHARGES_TO_DUMMY_ID"],
        row["PRESENT_DISCHARGE_PERCENTAGE"],
    ]


def build_sewershed_map(
    facilities_df, discharges_df, discharge_facility_lookup
):
    """
    Build sewershed map from facilities and discharges data

    Args:
        facilities_df: DataFrame containing facility information
        discharges_df: DataFrame containing discharge information

    Returns:
        Dictionary containing sewershed mapping data
    """

    sewershed_map = {}
    nodes_already_mapped = []

    # Add connections based on rows of DISCHARGES.csv
    print(f"Processing {len(discharges_df)} discharge connections")
    processed_connections = 0
    nodes_already_mapped = set()  # for lookup of mapped nodes
    for _, row in discharges_df.iterrows():
        processed_connections += 1
        if processed_connections % 2000 == 0:
            print(
                f"Processed {processed_connections} connections, found {len(sewershed_map)} sewersheds"
            )

        discharge_from_id = row["DUMMY_ID"]
        discharges_to = row["DISCHARGES_TO_DUMMY_ID"]
        if pd.isna(discharge_from_id) or pd.isna(discharges_to):
            continue  # Skip if either ID is NA. This will be handled by dummy ID

        if (
            discharge_from_id not in nodes_already_mapped
            and discharges_to not in nodes_already_mapped
        ):
            # Create new sewershed
            new_sewershed_id = len(sewershed_map) + 1
            sewershed_map[new_sewershed_id] = {
                "nodes": set([discharge_from_id, discharges_to]),
                "connections": [add_connection(row)],
            }
            nodes_already_mapped.update([discharge_from_id, discharges_to])
        else:
            # Add to existing sewershed
            for sewershed_info in sewershed_map.values():
                if (
                    discharge_from_id in sewershed_info["nodes"]
                    or discharges_to in sewershed_info["nodes"]
                ):
                    sewershed_info["nodes"].update(
                        [discharge_from_id, discharges_to]
                    )
                    sewershed_info["connections"].append(add_connection(row))
                    nodes_already_mapped.update(
                        [discharge_from_id, discharges_to]
                    )
                    break

    # Consolidate sewersheds with redundant nodes
    print(
        f"{len(sewershed_map)} sewersheds before combining sewersheds w/ repetitive nodes"
    )
    node_to_sewershed = {}
    sewershed_ids = list(sewershed_map.keys())

    for sewershed_id in sewershed_ids:
        if sewershed_id not in sewershed_map:  # Skip if already merged/deleted
            continue

        sewershed_info = sewershed_map[sewershed_id]
        for node in sewershed_info["nodes"]:
            if node in node_to_sewershed:  # Found overlapping node
                existing_sewershed_id = node_to_sewershed[
                    node
                ]  # Merge sewersheds
                if (
                    existing_sewershed_id != sewershed_id
                ):  # Into existing sewershed ID
                    sewershed_map[existing_sewershed_id]["nodes"].update(
                        sewershed_info["nodes"]
                    )
                    sewershed_map[existing_sewershed_id]["connections"].extend(
                        sewershed_info["connections"]
                    )
                    # Update node_to_sewershed for all nodes in the merged sewershed
                    for merged_node in sewershed_info["nodes"]:
                        node_to_sewershed[merged_node] = existing_sewershed_id
                    del sewershed_map[
                        sewershed_id
                    ]  # Remove duplicate sewershed
                    break
            else:
                node_to_sewershed[node] = sewershed_id

    print(
        f"{len(sewershed_map)} sewersheds after combining sewersheds w/ repetitive nodes"
    )

    discharge_location_data = discharges_df.set_index("DUMMY_ID")[
        ["STATE_CODE", "COUNTY_NAME"]
    ].to_dict("index")

    for dummy_id, location_info in discharge_location_data.items():
        if dummy_id in discharge_facility_lookup:
            discharge_facility_lookup[dummy_id].update(location_info)
        else:
            discharge_facility_lookup[dummy_id] = location_info

    new_sewershed_map = {}
    state_county_used = {}
    for sewershed_id, sewershed_info in sewershed_map.items():
        # Get location info for nodes
        node_info = []
        for node in sewershed_info["nodes"]:
            if node in discharge_facility_lookup:
                node_info.append(discharge_facility_lookup[node])

        # Get primary state and county
        if len(node_info) > 0:
            # Use Counter for faster counting
            state_counts = Counter(info["STATE_CODE"] for info in node_info)
            primary_state = (
                state_counts.most_common(1)[0][0]
                if state_counts
                else "Unspecified"
            )
            county_counts = Counter(
                info["COUNTY_NAME"]
                for info in node_info
                if info["STATE_CODE"] == primary_state
            )
            primary_county = (
                county_counts.most_common(1)[0][0]
                if county_counts
                else "Unspecified"
            )
        else:
            primary_state = "Unspecified"
            primary_county = "Unspecified"

        # Create new sewershed name
        state_county_key = f"{primary_state}_{primary_county}"
        state_county_used[state_county_key] = (
            state_county_used.get(state_county_key, 0) + 1
        )
        new_name = f"{primary_state} - {primary_county} County Sewershed {state_county_used[state_county_key]}"

        # Add node data using pre-computed lookup
        facility_columns = get_columns_from_configs(
            include_facility_base=True, include_calculated=False
        )
        node_data = {}
        for node in sewershed_info["nodes"]:
            node_data[node] = {}
            facility = discharge_facility_lookup.get(node)

            for key in facility_columns:
                if key == "DUMMY_ID":
                    node_data[node][
                        key
                    ] = node  # Use the node itself as DUMMY_ID
                else:
                    node_data[node][key] = (
                        facility.get(key) if facility else None
                    )

            node_data[node]["color"] = (
                get_node_color(
                    facility["FACILITY_TYPE"], facility["FACILITY_NAME"]
                )
                if facility and "FACILITY_TYPE" in facility
                else DEFAULT_NODE_COLOR
            )

        sewershed_info["node_data"] = node_data

        # Calculate aggregated sewershed population data for 2022 and 2042
        total_pop_2022 = 0
        total_pop_2042 = 0
        for node, data in node_data.items():
            if data.get("TOTAL_RES_POPULATION_2022"):
                total_pop_2022 += (
                    float(data["TOTAL_RES_POPULATION_2022"])
                    if data["TOTAL_RES_POPULATION_2022"]
                    else 0
                )
            if data.get("TOTAL_RES_POPULATION_2042"):
                total_pop_2042 += (
                    float(data["TOTAL_RES_POPULATION_2042"])
                    if data["TOTAL_RES_POPULATION_2042"]
                    else 0
                )

        sewershed_info["total_population_2022"] = total_pop_2022
        sewershed_info["total_population_2042"] = total_pop_2042
        sewershed_info["population_percent_increase"] = (
            ((total_pop_2042 - total_pop_2022) / total_pop_2022 * 100)
            if total_pop_2022 > 0
            else 0
        )

        new_sewershed_map[new_name] = sewershed_info
    print("Finished building sewershed map")
    sewershed_map = new_sewershed_map

    return sewershed_map


def update_external_discharges(
    discharges_df,
    facilities_df,
    facility_type_mappings,
):
    """Update external discharges involving facilities with multiple types"""
    # Create lookups for efficiency
    facility_type_lookup = facilities_df.set_index("CWNS_ID")[
        "FACILITY_TYPE"
    ].to_dict()
    facility_name_lookup = facilities_df.set_index("DUMMY_ID")[
        "FACILITY_NAME"
    ].to_dict()

    external_mask = (
        discharges_df["CWNS_ID"] != discharges_df["DISCHARGES_TO_CWNSID"]
    )

    for index, row in discharges_df[external_mask].iterrows():
        cwns_id, discharge_to_id = row["CWNS_ID"], row["DISCHARGES_TO_CWNSID"]
        discharge_type = row["DISCHARGE_TYPE"].lower()
        target_dummy_id = row["DISCHARGES_TO_DUMMY_ID"]

        # Case 1: Collection systems discharging to facility
        if (
            discharge_to_id in facility_type_mappings["collection"]
            and "collection" in facility_type_lookup.get(cwns_id, "").lower()
        ):
            discharges_df.loc[index, "DISCHARGES_TO_DUMMY_ID"] = (
                facility_type_mappings["collection"][discharge_to_id]
            )

        # Case 2: Facility discharging to reuse end-uses
        elif cwns_id in facility_type_mappings["reuse"] and any(
            term in discharge_type
            for term in ["reuse", "reclaim", "recycle", "pure"]
        ):
            discharges_df.loc[index, "DUMMY_ID"] = facility_type_mappings[
                "reuse"
            ][cwns_id]

        # Case 3: Facilities discharging to a separate pure water facility
        elif (
            cwns_id in facility_type_mappings["reuse"]
            and target_dummy_id in facility_name_lookup
            and any(
                term in facility_name_lookup[target_dummy_id].lower()
                for term in ["reuse", "pure"]
            )
        ):
            discharges_df.loc[index, "DUMMY_ID"] = facility_type_mappings[
                "reuse"
            ][cwns_id]

        # Case 4: Collection discharging to treatment plant
        elif (
            cwns_id in facility_type_mappings["collection"]
            and facility_type_lookup.get(discharge_to_id) == "Treatment Plant"
        ):
            discharges_df.loc[index, "DUMMY_ID"] = facility_type_mappings[
                "collection"
            ][cwns_id]

        # Case 5: Discharge to treatment plant
        elif discharge_to_id in facility_type_mappings["treatment_plant"]:
            discharges_df.loc[index, "DISCHARGES_TO_DUMMY_ID"] = (
                facility_type_mappings["treatment_plant"][discharge_to_id]
            )

        # Case 6: Discharge from treatment plant
        elif cwns_id in facility_type_mappings["treatment_plant"]:
            discharges_df.loc[index, "DUMMY_ID"] = facility_type_mappings[
                "treatment_plant"
            ][cwns_id]


def load_and_merge_cwns_data(
    data_dir="data/2022CWNS_NATIONAL_APR2024/", state=None
):
    """
    Load and merge CWNS data for a specific state or all states.

    Args:
        data_dir: Directory containing CWNS data files
        state: Optional state code to filter data (e.g., 'CA' for California)

    Returns:
        DataFrame containing merged CWNS facilities data with all related information
    """
    # Load data
    data_dict = load_cwns_data(data_dir)
    facilities = data_dict["facilities"]

    # Filter by state if specified
    if state:
        facilities = facilities[facilities["STATE_CODE"] == state]

    # Define which columns to keep from each dataframe
    merge_columns = {
        "permits": ["CWNS_ID", "PERMIT_NUMBER"],
        "counties": ["CWNS_ID", "COUNTY_NAME"],
        "types": ["CWNS_ID", "FACILITY_TYPE"],
        "flow": ["CWNS_ID", "CURRENT_DESIGN_FLOW"],
        "population": [
            "CWNS_ID",
            "TOTAL_RES_POPULATION_2022",
            "TOTAL_RES_POPULATION_2042",
        ],
        "physical_location": ["CWNS_ID", "LATITUDE", "LONGITUDE", "CITY"],
    }

    # Merge all dataframes
    for df_name, columns in merge_columns.items():
        df = data_dict[df_name][columns]
        facilities = facilities.merge(df, on="CWNS_ID", how="left")

    # Clean permit numbers
    facilities = clean_permit_numbers(facilities)

    for key in ["LATITUDE", "LONGITUDE"]:
        facilities[key] = pd.to_numeric(facilities[key], errors="coerce")

    return facilities


def main(
    data_dir="data/2022CWNS_NATIONAL_APR2024/",
    output_dir="processed_data/",
    state=None,
):
    """
    Main function to process CWNS data and create sewershed map

    Args:
        data_dir: Directory containing input CWNS data files
        output_dir: Directory for output files
        state: Optional state code to filter data (e.g., 'CA' for California)
    """

    # 1: Load and merge CWNS facility and discharges data
    facilities_2022 = load_and_merge_cwns_data(data_dir, state)

    # 2: Load discharges data
    discharges = pd.read_csv(
        f"{data_dir}DISCHARGES.csv", encoding="latin1", low_memory=False
    )

    # Filter discharges by state if specified
    if state:
        state_facilities = set(facilities_2022["CWNS_ID"].unique())
        discharges = discharges[
            (discharges["CWNS_ID"].isin(state_facilities))
            | (discharges["DISCHARGES_TO_CWNSID"].isin(state_facilities))
        ]
        print(f"{len(discharges)} discharge connections in {state}")

    # Add COUNTY_NAME to discharges via lookup from facilities
    county_lookup = facilities_2022[
        ["CWNS_ID", "COUNTY_NAME"]
    ].drop_duplicates()
    discharges = discharges.merge(county_lookup, on="CWNS_ID", how="left")

    # 3: Preprocess facilities and discharges
    print("Preprocessing facilities and discharges")
    facilities_2022, discharges = preprocess_facilities_and_discharges(
        facilities_2022, discharges
    )

    # Create discharge facility lookup after preprocessing and adding new dummy IDs
    discharge_facility_lookup = facilities_2022.set_index("DUMMY_ID").to_dict(
        "index"
    )

    # Handle duplicate CWNS_IDs by keeping the first occurrence
    facilities_unique_cwns = facilities_2022.drop_duplicates(
        subset="CWNS_ID", keep="first"
    )
    cwns_facility_lookup = facilities_unique_cwns.set_index("CWNS_ID").to_dict(
        "index"
    )

    discharge_location_data = discharges.set_index("DUMMY_ID")[
        ["STATE_CODE", "COUNTY_NAME", "CWNS_ID", "DISCHARGE_TYPE"]
    ].to_dict("index")

    for dummy_id, location_info in discharge_location_data.items():
        cwns_id = location_info["CWNS_ID"]

    for dummy_id, location_info in discharge_location_data.items():
        cwns_id = location_info["CWNS_ID"]
        # Copy facility info and add discharge-specific location info
        discharge_facility_lookup[dummy_id] = cwns_facility_lookup[
            cwns_id
        ].copy()

        # Map discharge types to facility types based on facility_type_groups.json
        discharge_type = str(location_info["DISCHARGE_TYPE"]).lower()
        if "ocean" in discharge_type or "marine" in discharge_type:
            facility_type = "Ocean Discharge"
        elif (
            "reuse" in discharge_type
            or "reclamation" in discharge_type
            or "land application" in discharge_type
            or "irrigation" in discharge_type
            or "groundwater" in discharge_type
            or "injection" in discharge_type
        ):
            facility_type = "Reuse"
        else:
            facility_type = "Other"

        discharge_facility_lookup[dummy_id].update(
            {
                "STATE_CODE": location_info["STATE_CODE"],
                "COUNTY_NAME": location_info["COUNTY_NAME"],
                "FACILITY_TYPE": facility_type,
            }
        )

    # 4: Build sewershed map
    print("Building sewershed map")
    sewershed_map = build_sewershed_map(
        facilities_2022, discharges, discharge_facility_lookup
    )

    # Calculate population percent increase, handling division by zero
    pop_2022 = facilities_2022["TOTAL_RES_POPULATION_2022"].astype(float)
    pop_2042 = facilities_2022["TOTAL_RES_POPULATION_2042"].astype(float)

    # Avoid division by zero by replacing 0 with 1
    pop_2022_safe = pop_2022.replace(0, 1)

    facilities_2022["POPULATION_PERCENT_INCREASE"] = (
        (pop_2042 - pop_2022) / pop_2022_safe * 100
    ).round(2)

    # Handle cases where 2022 population was 0 - set increase to 0
    zero_pop_mask = pop_2022 == 0
    facilities_2022.loc[zero_pop_mask, "POPULATION_PERCENT_INCREASE"] = 0
    # Replace infinite values and NaN with 0
    facilities_2022["POPULATION_PERCENT_INCREASE"] = (
        facilities_2022["POPULATION_PERCENT_INCREASE"]
        .replace([float("inf"), -float("inf")], 0)
        .fillna(0)
    )

    print("Saving outputs")
    output_prefix = f"{state}_" if state else ""
    output_columns = get_columns_from_configs(
        include_facility_base=True, include_calculated=True
    )
    facilities_2022[output_columns].to_csv(
        f"{output_dir}{output_prefix}facilities_2022_merged.csv", index=False
    )
    with open(f"{output_dir}{output_prefix}sewershed_map.pkl", "wb") as f:
        pickle.dump(sewershed_map, f)


if __name__ == "__main__":
    # main(state="CA")
    main()
