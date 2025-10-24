import os
import json


# Load configuration files from JSON
def load_json(file_path):
    """Load data from JSON file"""
    full_path = os.path.join(os.path.dirname(__file__), file_path)
    with open(full_path, "r") as f:
        return json.load(f)


FACILITY_TYPE_GROUPS = load_json("facility_type_groups.json")
FILE_CONFIGS = load_json("file_configs.json")
CONNECTION_RULES = load_json("connection_rules.json")


all_columns = {"CWNS_ID", "FACILITY_NAME"}
for config in FILE_CONFIGS.values():
    columns = config["columns"].copy()
    if "rename" in config:  # Use renamed version of key
        for old_name, new_name in config["rename"].items():
            columns[columns.index(old_name)] = new_name
    if "filter" in config:  # Remove keys used only to filter
        filter_keys = set(config["filter"].keys())
        columns = [col for col in columns if col not in filter_keys]
    all_columns.update(columns)
    all_columns.add("POP_PERCENT_INCREASE")
    all_columns -= {
        "DISCHARGES_TO_CWNSID",
        "DISCHARGE_TYPE",
        "PRESENT_DISCHARGE_PERCENTAGE",
    }
OUTPUT_COLUMNS = sorted(all_columns)
TYPE_SPECIFIC_COLUMNS = ["FACILITY_TYPE", "LATITUDE", "LONGITUDE"]


FACILITY_TYPE_ORDER = {}
FACILITY_TYPE_TO_GROUP = {}
DISCHARGE_TYPE_TO_FACILITY_TYPE = {}
for group_name, group_data in FACILITY_TYPE_GROUPS.items():
    processing_order = group_data.get("order", 999)
    for facility_type in group_data["TYPE_LIST"]:
        FACILITY_TYPE_ORDER[facility_type] = processing_order
        FACILITY_TYPE_TO_GROUP[facility_type] = group_name
    if "discharge_type_keywords" in group_data:
        type = group_data["TYPE_LIST"][0]  # first facility type from group
        for keyword in group_data["discharge_type_keywords"]:
            DISCHARGE_TYPE_TO_FACILITY_TYPE[keyword.lower()] = type


def get_coords(facility):
    """Extract coordinates from a facility"""
    return (facility["LATITUDE"], facility["LONGITUDE"])


def add_facility_type_node(facility_type, base_coords, index):
    """Create a standardized facility type entry for network graph"""
    color, shape = get_node_style(facility_type)
    return {
        "FACILITY_TYPE": facility_type,
        "color": color,
        "shape": shape,
    }


def get_node_style(facility_type):
    """Get color and shape for facility type based on groups"""
    for group_data in FACILITY_TYPE_GROUPS.values():
        if facility_type in group_data["TYPE_LIST"]:
            return group_data["color"], group_data["shape"]
    return "#FFFFC5", "ellipse"  # Default light yellow, ellipse


def get_facility_types(group_name_or_names):
    """Get all facility types in one or more groups"""
    if isinstance(group_name_or_names, str):
        return FACILITY_TYPE_GROUPS[group_name_or_names]["TYPE_LIST"]
    else:
        return [
            fac_type
            for group_name in group_name_or_names
            for fac_type in FACILITY_TYPE_GROUPS[group_name]["TYPE_LIST"]
        ]
