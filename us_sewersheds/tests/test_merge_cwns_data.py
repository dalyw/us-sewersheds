import os
import pytest
import pandas as pd
from us_sewersheds.merge_cwns_data import load_cwns_data, load_and_merge_cwns_data, build_sewershed_map

# set skip_all_tests = True to focus on single test
skip_all_tests = False

@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "data_dir, expected_keys",
    [
        (
            "data/2022CWNS_NATIONAL_APR2024/",
            ["facilities", "permits", "counties", "types", "flow"]
        ),
        (
            "us_sewersheds/tests/data/",
            ["facilities", "permits", "counties", "types", "flow"]
        ),
    ],
)
def test_load_cwns_data(data_dir, expected_keys):
    """Test that load_cwns_data returns expected data structure with correct keys."""
    result = load_cwns_data(data_dir)
    
    assert isinstance(result, dict)
    assert all(key in result for key in expected_keys)
    
    # Check that all values are DataFrames
    for key in expected_keys:
        assert isinstance(result[key], pd.DataFrame)
        assert not result[key].empty, f"DataFrame for {key} should not be empty"
    
    facilities = result["facilities"]
    assert "CWNS_ID" in facilities.columns
    assert "FACILITY_NAME" in facilities.columns
    assert "STATE_CODE" in facilities.columns
    
    permits = result["permits"]
    assert "PERMIT_NUMBER" in permits.columns
    
    counties = result["counties"]
    assert "COUNTY_NAME" in counties.columns


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "data_dir, state, expected_columns",
    [
        (
            "data/2022CWNS_NATIONAL_APR2024/",
            None,
            ["CWNS_ID", "FACILITY_NAME", "STATE_CODE"]
        ),
        (
            "data/2022CWNS_NATIONAL_APR2024/",
            "CA",
            ["CWNS_ID", "FACILITY_NAME", "STATE_CODE"]
        ),
        (
            "us_sewersheds/tests/data/",
            None,
            ["CWNS_ID", "FACILITY_NAME", "STATE_CODE"]
        ),
        (
            "us_sewersheds/tests/data/",
            "CA",
            ["CWNS_ID", "FACILITY_NAME", "STATE_CODE"]
        ),
    ],
)
def test_load_and_merge_cwns_data(data_dir, state, expected_columns):
    """Test that load_and_merge_cwns_data returns merged DataFrame with expected structure."""
    result = load_and_merge_cwns_data(data_dir, state)
    
    # Check that result is a DataFrame
    assert isinstance(result, pd.DataFrame)
    
    for col in expected_columns:
        assert col in result.columns, f"Column {col} should be present in merged data"
    
    if state:
        assert all(result["STATE_CODE"] == state), f"All facilities should be in state {state}"
        
    # Check that there are no null values in key columns
    assert not result["CWNS_ID"].isnull().any(), "CWNS_ID should not have null values"
    assert not result["FACILITY_NAME"].isnull().any(), "FACILITY_NAME should not have null values"


def create_test_facilities_df():
    """Helper function to create test facilities DataFrame with DUMMY_ID."""
    facilities_df = pd.read_csv("us_sewersheds/tests/data/FACILITIES.csv")
    facilities_df["DUMMY_ID"] = facilities_df["CWNS_ID"].astype(str)
    # Convert PERMIT_NUMBER to list format as expected by the function
    facilities_df["PERMIT_NUMBER"] = facilities_df["PERMIT_NUMBER"].apply(lambda x: [x])
    return facilities_df

def create_test_discharges_df(test_case):
    """Helper function to create test discharges DataFrame based on test case."""
    if test_case == "basic_connections":
        discharges_df = pd.DataFrame({
            'CWNS_ID': [1002, 1005],
            'DUMMY_ID': ['1002', '1005'],
            'DISCHARGES_TO_CWNSID': [1001, 1003],
            'DISCHARGES_TO_DUMMY_ID': ['1001', '1003'],
            'DISCHARGE_TYPE': ['Collection to Treatment', 'Collection to Treatment'],
            'PRESENT_DISCHARGE_PERCENTAGE': [100, 100],
            'STATE_CODE': ['CA', 'TX'],
            'COUNTY_NAME': ['Test County', 'Sample County']
        })
    elif test_case == "no_connections":
        discharges_df = pd.DataFrame({
            'CWNS_ID': [],
            'DUMMY_ID': [],
            'DISCHARGES_TO_CWNSID': [],
            'DISCHARGES_TO_DUMMY_ID': [],
            'DISCHARGE_TYPE': [],
            'PRESENT_DISCHARGE_PERCENTAGE': [],
            'STATE_CODE': [],
            'COUNTY_NAME': []
        })
    elif test_case == "single_connection":
        discharges_df = pd.DataFrame({
            'CWNS_ID': [1002],
            'DUMMY_ID': ['1002'],
            'DISCHARGES_TO_CWNSID': [1001],
            'DISCHARGES_TO_DUMMY_ID': ['1001'],
            'DISCHARGE_TYPE': ['Collection to Treatment'],
            'PRESENT_DISCHARGE_PERCENTAGE': [100],
            'STATE_CODE': ['CA'],
            'COUNTY_NAME': ['Test County']
        })
    return discharges_df

@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "test_case, expected_connections",
    [
        (
            "basic_connections",
            [("1002", "1001"), ("1005", "1003")]
        ),
        (
            "no_connections", 
            []
        ),
        (
            "single_connection",
            [("1002", "1001")]
        ),
    ],
)
def test_build_sewershed_map(test_case, expected_connections):
    """Test that build_sewershed_map returns expected sewershed mapping structure."""
    
    # Load test data
    facilities_df = create_test_facilities_df()
    discharges_df = create_test_discharges_df(test_case)
    
    # Test the function
    result = build_sewershed_map(facilities_df, discharges_df)
    
    # Check that result is a dictionary
    assert isinstance(result, dict), "Result should be a dictionary"
    
    # Check that we have some sewersheds (unless no connections case)
    if expected_connections:
        assert len(result) > 0, "Should have at least one sewershed"
    else:
        # For no_connections case, we might have 0 sewersheds
        assert len(result) >= 0, "Should have zero or more sewersheds"
    
    # Check structure of each sewershed entry
    for sewershed_id, sewershed_data in result.items():
        assert isinstance(sewershed_data, dict), f"Sewershed {sewershed_id} should be a dictionary"
        assert 'nodes' in sewershed_data, f"Sewershed {sewershed_id} should have 'nodes' key"
        assert 'connections' in sewershed_data, f"Sewershed {sewershed_id} should have 'connections' key"
        assert isinstance(sewershed_data['nodes'], set), f"Sewershed {sewershed_id} nodes should be a set"
        assert isinstance(sewershed_data['connections'], list), f"Sewershed {sewershed_id} connections should be a list"
    
    # Check expected connections
    if expected_connections:
        for from_id, to_id in expected_connections:
            has_connection = any(
                any(conn[0] == from_id and conn[1] == to_id for conn in sewershed['connections'])
                for sewershed in result.values()
            )
            assert has_connection, f"Should have connection from {from_id} to {to_id}"
    else:
        # For no_connections case, verify no connections exist
        total_connections = sum(len(sewershed['connections']) for sewershed in result.values())
        assert total_connections == 0, "Should have no connections for no_connections test case"
