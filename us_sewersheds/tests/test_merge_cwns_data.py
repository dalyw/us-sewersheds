import pytest
import pandas as pd
from us_sewersheds.merge_cwns_data import (
    load_and_merge_cwns_data,
    process_multi_type_facilities,
    create_sewershed_map,
)

# set skip_all_tests = True to focus on single test
skip_all_tests = False


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
@pytest.mark.parametrize(
    "data_dir, state, expected_columns",
    [
        (
            "data/2022CWNS_NATIONAL_APR2024/",
            None,
            ["CWNS_ID", "FACILITY_NAME", "STATE_CODE"],
        ),
        (
            "data/2022CWNS_NATIONAL_APR2024/",
            "CA",
            ["CWNS_ID", "FACILITY_NAME", "STATE_CODE"],
        ),
        (
            "us_sewersheds/tests/data/",
            None,
            ["CWNS_ID", "FACILITY_NAME", "STATE_CODE"],
        ),
        (
            "us_sewersheds/tests/data/",
            "CA",
            ["CWNS_ID", "FACILITY_NAME", "STATE_CODE"],
        ),
    ],
)
def test_load_and_merge_cwns_data(data_dir, state, expected_columns):
    """Test that load_and_merge_cwns_data returns
    merged data with expected structure."""
    result = load_and_merge_cwns_data(data_dir, state)

    assert isinstance(result, dict)
    assert "FACILITIES" in result
    assert "DISCHARGES" in result

    facilities = result["FACILITIES"]
    discharges = result["DISCHARGES"]

    assert isinstance(facilities, pd.DataFrame)
    assert isinstance(discharges, pd.DataFrame)

    for col in expected_columns:
        assert col in facilities.columns

    if state:
        assert all(facilities["STATE_CODE"] == state)

    # Check that there are no null values in key columns
    assert not facilities["CWNS_ID"].isnull().any()
    assert not facilities["FACILITY_NAME"].isnull().any()


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_process_multi_type_facilities():
    """Test that process_multi_type_facilities returns expected structure."""
    # Load test data
    data = load_and_merge_cwns_data("us_sewersheds/tests/data/")

    result = process_multi_type_facilities(data)

    assert isinstance(result, dict)
    assert len(result) > 0

    # Check structure of first facility
    first_facility = list(result.values())[0]
    assert "TYPES" in first_facility
    assert isinstance(first_facility["TYPES"], dict)

    # Check that each type has required keys
    for type_data in first_facility["TYPES"].values():
        assert "FACILITY_TYPE" in type_data
        assert "color" in type_data
        assert "shape" in type_data


@pytest.mark.skipif(skip_all_tests, reason="Exclude all tests")
def test_create_sewershed_map():
    """Test that create_sewershed_map returns expected structure."""
    # Load test data
    data = load_and_merge_cwns_data("us_sewersheds/tests/data/")
    processed_facilities = process_multi_type_facilities(data)

    result = create_sewershed_map(data, processed_facilities)

    assert isinstance(result, dict)

    # Check structure of sewersheds
    for sewershed_id, sewershed_data in result.items():
        assert isinstance(sewershed_data, dict)
        assert "nodes" in sewershed_data
        assert "connections" in sewershed_data
        assert isinstance(sewershed_data["nodes"], dict)
        assert isinstance(sewershed_data["connections"], list)

        # Check node structure
        for node_id, node_data in sewershed_data["nodes"].items():
            assert "TYPES" in node_data
            assert isinstance(node_data["TYPES"], dict)
