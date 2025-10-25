import pandas as pd
import json

from us_sewersheds.helpers import (
    FILE_CONFIGS,
    CONNECTION_RULES,
    FACILITY_TYPE_GROUPS,
    OUTPUT_COLUMNS,
    FACILITY_TYPE_ORDER,
    DISCHARGE_TYPE_TO_FACILITY_TYPE,
    TYPE_SPECIFIC_COLUMNS,
    add_facility_type_node,
    get_facility_types,
    get_coords,
)


def add_internal_connections(node_id, TYPES, sewershed):
    """Internal connection rules for facility types within a node"""

    # Sequential treatment connections
    sequential_rule = CONNECTION_RULES["internal"]["sequential"]
    max_order = sequential_rule["max_processing_order"]
    treatment_types = []
    for i, facility_type in TYPES.items():
        fac_type = facility_type["FACILITY_TYPE"]
        if FACILITY_TYPE_ORDER[fac_type] <= max_order:
            treatment_types.append((i, FACILITY_TYPE_ORDER[fac_type]))
    treatment_types.sort(key=lambda x: x[1])
    for i in range(len(treatment_types) - 1):
        source_idx = treatment_types[i][0]
        target_idx = treatment_types[i + 1][0]
        source_id = f"{node_id}_type_{source_idx}"
        target_id = f"{node_id}_type_{target_idx}"
        connection = (source_id, target_id, 100)
        sewershed["connections"].add(connection)

    # Diverging discharge connections
    if treatment_types:
        diverging_rule = CONNECTION_RULES["internal"]["diverging"]
        last_treatment_idx = treatment_types[-1][0]
        last_treatment_id = f"{node_id}_type_{last_treatment_idx}"
        target_types = get_facility_types(diverging_rule["target_groups"])
        for i, facility_type in TYPES.items():
            fac_type = facility_type["FACILITY_TYPE"]
            if fac_type in target_types:
                discharge_id = f"{node_id}_type_{i}"
                connection = (last_treatment_id, discharge_id, None)
                sewershed["connections"].add(connection)


def get_connection_indices(source, target):
    """Get source and target facility type indices for external connections"""
    # Apply source selection rules
    source_idx = len(source["TYPES"]) - 1  # Default: last type
    source_types = [ft["FACILITY_TYPE"] for ft in source["TYPES"].values()]
    target_types = [ft["FACILITY_TYPE"] for ft in target["TYPES"].values()]
    for source_group, rule in CONNECTION_RULES["rules"]["external"].items():
        # Get facility types for source and target groups
        source_group_types = get_facility_types(source_group)
        target_group_types = get_facility_types(rule["target_conditions"])
        if all(
            req_type in source_types for req_type in source_group_types
        ) and all(req_type in target_types for req_type in target_group_types):
            # Find the specified source type index (use first type from group)
            for i, facility_type in enumerate(source["TYPES"].values()):
                if facility_type["FACILITY_TYPE"] in source_group_types:
                    source_idx = i
                    break
            break

    # Apply target selection rules
    target_idx = 0  # Default: first type
    min_order = CONNECTION_RULES["rules"]["target"]["min_processing_order"]

    for i, facility_type in enumerate(target["TYPES"].values()):
        fac_type = facility_type["FACILITY_TYPE"]
        group_order = FACILITY_TYPE_ORDER[fac_type]
        if group_order >= min_order:
            target_idx = i
            break

    return source_idx, target_idx


def load_and_merge_cwns_data(data_dir, state=None, data={}):
    """Load and merge CWNS data for a specific state or all states."""

    # Load each file
    for filename, config in FILE_CONFIGS.items():
        kwargs = {"encoding": "latin1", "low_memory": False}
        df = pd.read_csv(f"{data_dir}{filename}", **kwargs)[config["columns"]]

        for col, value in config.get("filter", {}).items():
            df = df[df[col] == value]

        df = df.drop(config.get("drop", []), axis=1)

        if config.get("groupby"):
            df = (
                df.groupby(config["groupby"])
                .agg(lambda x: "; ".join(x.dropna().astype(str)))
                .reset_index()
            )

        data[filename.replace(".csv", "")] = df

    # Filter by state if specified
    if state:
        for key in ["FACILITIES", "DISCHARGES"]:
            if "STATE_CODE" in data[key].columns:
                data[key] = data[key][data[key]["STATE_CODE"] == state]
        print(f"{len(data['DISCHARGES'])} discharge connections in {state}")

    # Merge dataframes into FACILITIES
    for filename, config in FILE_CONFIGS.items():
        df_name = filename.replace(".csv", "")
        if df_name not in ["FACILITIES", "DISCHARGES"]:
            print(f"Merging {df_name}")
            if df_name == "FACILITY_TYPES":
                # Merge facility types directly to preserve multiple types per facility
                data["FACILITIES"] = data["FACILITIES"].merge(
                    data[df_name], on="CWNS_ID", how="left"
                )
            else:
                # Check for overlapping columns and handle duplicates
                overlapping_cols = set(data["FACILITIES"].columns) & set(
                    data[df_name].columns
                )
                overlapping_cols.discard("CWNS_ID")  # Merge key
                df_to_merge = data[df_name].drop(columns=list(overlapping_cols))
                data["FACILITIES"] = data["FACILITIES"].merge(
                    df_to_merge, on="CWNS_ID", how="left"
                )

    # Merge COUNTY_NAME to discharges
    county_data = data["FACILITIES"][["CWNS_ID", "COUNTY_NAME"]]
    data["DISCHARGES"] = data["DISCHARGES"].merge(
        county_data, on="CWNS_ID", how="left"
    )

    # Calculate population percent increase
    pop_2022 = data["FACILITIES"]["TOTAL_RES_POPULATION_2022"]
    pop_2042 = data["FACILITIES"]["TOTAL_RES_POPULATION_2042"]
    data["FACILITIES"]["POP_PERCENT_INCREASE"] = (
        ((pop_2042 - pop_2022) / pop_2022 * 100)
        .where(pop_2022 != 0, pd.NA)
        .round(2)
    )

    data["FACILITIES"] = clean_permit_numbers(data["FACILITIES"])
    return {
        "FACILITIES": clean_cwns_ids(data["FACILITIES"]),
        "DISCHARGES": clean_cwns_ids(data["DISCHARGES"]),
    }


def clean_cwns_ids(df):
    """Clean and convert CWNS_IDs to integers"""
    df = df.copy()
    cwns_columns = ["CWNS_ID", "DISCHARGES_TO_CWNSID"]
    for col in cwns_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def clean_permit_numbers(facilities_df):
    # remove common patterns with case-insensitive matching
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


def _merge_sewersheds(sewersheds):
    """Merge sewersheds that are connected by external discharges"""
    print("Merging connected sewersheds")
    merged_ssheds = {}
    processed_ssheds = set()

    for ss_key, sewershed in sewersheds.items():
        if ss_key in processed_ssheds:
            continue

        # Find all connected sewersheds
        merged_ss = sewershed.copy()
        to_process = [ss_key]
        processed_ssheds.add(ss_key)
        while to_process:
            current_key = to_process.pop(0)
            current_sshed = sewersheds[current_key]

            # Check all connections to find external connections
            for connection in list(current_sshed["connections"]):
                source_id, target_id, percentage = connection

                # Extract base IDs (remove _type_X suffix)
                source_base = source_id.split("_type_")[0]
                target_base = target_id.split("_type_")[0]

                # If external connection, find the target's sewershed
                if source_base != target_base:
                    for other_key, other_ss in sewersheds.items():
                        if (
                            other_key != current_key
                            and other_key not in processed_ssheds
                        ):
                            if target_base in other_ss["nodes"]:
                                # Merge the other sewershed
                                merged_ss["nodes"].update(other_ss["nodes"])
                                merged_ss["connections"].update(
                                    other_ss["connections"]
                                )

                                processed_ssheds.add(other_key)
                                to_process.append(other_key)
                                break

        merged_ssheds[ss_key] = merged_ss

    return merged_ssheds


def process_multi_type_facilities(data):
    """Process facilities with multiple types using nested structure"""
    print("Processing multi-type facilities")

    # Pre-filter discharges for external ones (vectorized)
    external_discharges = data["DISCHARGES"][
        data["DISCHARGES"]["DISCHARGE_TYPE"].notna()
        & data["DISCHARGES"]["DISCHARGES_TO_CWNSID"].isna()
    ].copy()

    # Group external discharges by CWNS_ID for faster lookup
    external_discharges_by_facility = external_discharges.groupby("CWNS_ID")[
        "DISCHARGE_TYPE"
    ].apply(set)

    processed_facilities = {}

    # Group facilities by CWNS_ID to find multi-type facilities
    facility_groups = data["FACILITIES"].groupby("CWNS_ID")
    id_count = 0

    for cwns_id, group in facility_groups:
        id_count += 1
        if id_count % 2500 == 0:
            print(f"{id_count} facilities processed")

        # Get all facility types for this CWNS_ID
        group_types = group["FACILITY_TYPE"].unique()
        sorted_types = sorted(group_types, key=lambda x: FACILITY_TYPE_ORDER[x])
        base_coords = get_coords(group.iloc[0])
        TYPES = {}

        # Add facility types
        for t, fac_type in enumerate(sorted_types):
            TYPES[t] = add_facility_type_node(fac_type, base_coords, t)

        # Add discharge types using direct dictionary lookup
        if cwns_id in external_discharges_by_facility:
            unique_discharge_types = set()
            for discharge_type in external_discharges_by_facility[cwns_id]:
                discharge_facility_type = DISCHARGE_TYPE_TO_FACILITY_TYPE.get(
                    discharge_type, "Other"
                )

                if discharge_facility_type not in unique_discharge_types:
                    unique_discharge_types.add(discharge_facility_type)
                    next_index = len(TYPES)
                    TYPES[next_index] = add_facility_type_node(
                        discharge_facility_type,
                        base_coords,
                        next_index,
                    )

        processed_facilities[str(cwns_id)] = {
            **group.iloc[0][OUTPUT_COLUMNS].to_dict(),
            "TYPES": TYPES,
        }

    # Batch update all external discharges at once (vectorized)
    external_mask = data["DISCHARGES"]["DISCHARGES_TO_CWNSID"].isna()
    data["DISCHARGES"].loc[external_mask, "DISCHARGES_TO_CWNSID"] = data[
        "DISCHARGES"
    ].loc[external_mask, "CWNS_ID"]

    print(f"Processed {len(processed_facilities)} facilities with nested types")
    return processed_facilities


def create_sewershed_map(data, processed_facilities):
    """
    Create sewershed map by grouping connected facilities.
    Uses nested structure with TYPES array.
    """
    print("Creating sewershed map")

    # Group discharges by connected facilities to create sewersheds
    sewersheds = {}

    for _, discharge in data["DISCHARGES"].iterrows():
        source_cwns_id = discharge["CWNS_ID"]
        target_cwns_id = discharge["DISCHARGES_TO_CWNSID"]

        if pd.isna(target_cwns_id) or source_cwns_id == target_cwns_id:
            continue

        # Find existing sewershed or create new one
        source_id = str(source_cwns_id)
        target_id = str(target_cwns_id)
        ss_key = None
        for key, sewershed in sewersheds.items():
            if (
                source_id in sewershed["nodes"]
                or target_id in sewershed["nodes"]
            ):
                ss_key = key
                break

        if ss_key is None:  # Create new sewershed
            ss_key = f"Sewershed_{len(sewersheds) + 1}"
            sewersheds[ss_key] = {"connections": set(), "nodes": {}}

        # Add nodes and connection to sewershed
        for id in [source_id, target_id]:
            if id not in sewersheds[ss_key]["nodes"]:
                if id not in processed_facilities:
                    # Add dummy "unknown" node
                    other_id = target_id if id == source_id else source_id
                    other_facility = processed_facilities[other_id]
                    processed_facilities[id] = {
                        **other_facility,
                        "CWNS_ID": int(id),
                        "FACILITY_NAME": f"Missing Facility {id}",
                        "TYPES": {
                            0: add_facility_type_node(
                                "Unknown", get_coords(other_facility), 0
                            )
                        },
                    }
                sewersheds[ss_key]["nodes"][id] = processed_facilities[id]
        source_type_idx, target_type_idx = get_connection_indices(
            processed_facilities[source_id], processed_facilities[target_id]
        )
        connection = [
            f"{source_id}_type_{source_type_idx}",
            f"{target_id}_type_{target_type_idx}",
            discharge.get("PRESENT_DISCHARGE_PERCENTAGE", 100),
        ]
        sewersheds[ss_key]["connections"].add(tuple(connection))

    # Merge connected sewersheds
    sewersheds = _merge_sewersheds(sewersheds)

    # Convert to final format
    final_sewersheds = {}
    county_counts = {}

    for key, sewershed in sewersheds.items():
        if len(sewershed["nodes"]) < 2:  # Skip single-node sewersheds
            continue

        # Get state/county from first node
        first_node_id = list(sewershed["nodes"].keys())[0]
        first_node = sewershed["nodes"].get(first_node_id, {})
        state_code = first_node.get("STATE_CODE", "Unknown")
        county_name = first_node.get("COUNTY_NAME", "Unknown")

        # Create unique name with count
        county_key = f"{state_code} - {county_name}"
        county_counts[county_key] = county_counts.get(county_key, 0) + 1
        ss_name = f"{county_key} Sewershed {county_counts[county_key]}"
        final_sewersheds[ss_name] = sewershed

    # Add internal connections between facility types based on processing order
    print("Adding internal facility type connections")
    for sewershed_name, sewershed in final_sewersheds.items():
        for node_id, node_data in sewershed["nodes"].items():
            if len(node_data["TYPES"]) > 1:  # has internal connections
                add_internal_connections(node_id, node_data["TYPES"], sewershed)

        # Convert set back to list for JSON serialization
        sewershed["connections"] = list(sewershed["connections"])

    print(f"Created {len(final_sewersheds)} sewersheds")
    return final_sewersheds


def main(
    data_dir="data/2022CWNS_NATIONAL_APR2024/",
    state=None,
):

    data = load_and_merge_cwns_data(data_dir, state)
    processed_facilities = process_multi_type_facilities(data)
    sewersheds = create_sewershed_map(data, processed_facilities)

    # Save facilities CSV (flatten nested structure for CSV)
    flattened_facilities = []
    for facility_id, facility_data in processed_facilities.items():
        base_data = {k: v for k, v in facility_data.items() if k != "TYPES"}

        for i, facility_type in facility_data["TYPES"].items():
            flattened_facility = base_data.copy()
            # Override keys that vary by facility type
            for key in TYPE_SPECIFIC_COLUMNS:
                flattened_facility[key] = facility_type.get(
                    key, facility_data.get(key, "")
                )
            if "DISCHARGE_TYPE" in facility_type:
                flattened_facility["DISCHARGE_TYPE"] = facility_type[
                    "DISCHARGE_TYPE"
                ]
            flattened_facilities.append(flattened_facility)
    facilities_df = pd.DataFrame(flattened_facilities)

    # Filter out facilities with any type from the "Other" category
    other_types = FACILITY_TYPE_GROUPS["Other"]["TYPE_LIST"]
    facilities_df = facilities_df[
        ~facilities_df["FACILITY_TYPE"].isin(other_types)
    ]
    prefix = f"{state}_" if state else ""
    facilities_df[OUTPUT_COLUMNS].to_csv(
        f"processed_data/{prefix}facilities_merged.csv", index=False
    )

    # Save sewershed map JSON
    with open(f"processed_data/{prefix}sewershed_map.json", "w") as f:
        json.dump(sewersheds, f, indent=2)
    print(f"Saved sewershed map with {len(sewersheds)} sewersheds")


if __name__ == "__main__":
    main()
