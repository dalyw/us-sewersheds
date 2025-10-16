"""
Spatial analysis functions for sewershed mapping and H3 hexagonal indexing.

Inspired by the USEPA Sewersheds repository
(https://github.com/USEPA/Sewersheds), adapting hexagonal grid methodology
for wastewater infrastructure analysis.
"""

import pandas as pd
import h3
import folium


def add_h3_indexing(facilities_df, resolution=8):
    """Add H3 hexagonal indexing to facilities based on coordinates.

    Credits: H3 methodology adapted from USEPA Sewersheds repository
    """
    facilities_df = facilities_df.copy()
    facilities_df["LATITUDE"] = pd.to_numeric(
        facilities_df["LATITUDE"], errors="coerce"
    )
    facilities_df["LONGITUDE"] = pd.to_numeric(
        facilities_df["LONGITUDE"], errors="coerce"
    )

    valid_coords = facilities_df.dropna(subset=["LATITUDE", "LONGITUDE"])
    valid_coords["H3_INDEX"] = valid_coords.apply(
        lambda row: h3.latlng_to_cell(row["LATITUDE"], row["LONGITUDE"], resolution),
        axis=1,
    )

    return valid_coords


def find_spatial_neighbors(facilities_df, k_ring=1):
    """Find spatially neighboring facilities using H3 hexagonal grid.

    Credits: H3 methodology adapted from USEPA Sewersheds repository
    """
    facilities_df = facilities_df.copy()
    facilities_df["H3_NEIGHBORS"] = facilities_df["H3_INDEX"].apply(
        lambda h3_index: list(h3.grid_ring(h3_index, k=k_ring))
    )
    return facilities_df


def aggregate_by_h3_hexagon(facilities_df):
    """Aggregate facility data by H3 hexagon for spatial clustering.

    Credits: H3 methodology adapted from USEPA Sewersheds repository
    """
    agg_functions = {
        "CWNS_ID": "count",
        "TOTAL_RES_POPULATION_2022": "sum",
        "CURRENT_DESIGN_FLOW": "sum",
        "LATITUDE": "mean",
        "LONGITUDE": "mean",
        "STATE_CODE": lambda x: (
            x.mode().iloc[0] if not x.empty and len(x.mode()) > 0 else None
        ),
        "COUNTY_NAME": lambda x: (
            x.mode().iloc[0] if not x.empty and len(x.mode()) > 0 else None
        ),
        "FACILITY_TYPE": lambda x: list(x.unique()),
    }

    spatial_clusters = (
        facilities_df.groupby("H3_INDEX").agg(agg_functions).reset_index()
    )
    spatial_clusters = spatial_clusters.rename(
        columns={
            "CWNS_ID": "FACILITY_COUNT",
            "LATITUDE": "CENTROID_LAT",
            "LONGITUDE": "CENTROID_LON",
        }
    )

    return spatial_clusters


def calculate_spatial_distances(facilities_df, max_distance_km=50.0):
    """Calculate spatial distances between
    facilities using H3-based approximation."""
    distances = []
    facilities_list = facilities_df[["CWNS_ID", "LATITUDE", "LONGITUDE"]].values

    for i, (cwns_id1, lat1, lon1) in enumerate(facilities_list):
        for j, (cwns_id2, lat2, lon2) in enumerate(facilities_list[i + 1 :], i + 1):
            if cwns_id1 != cwns_id2:
                distance_km = h3.latlng_distance((lat1, lon1), (lat2, lon2), unit="km")
                if distance_km <= max_distance_km:
                    distances.append(
                        {
                            "FROM_CWNS_ID": cwns_id1,
                            "TO_CWNS_ID": cwns_id2,
                            "DISTANCE_KM": distance_km,
                        }
                    )

    return pd.DataFrame(distances)


def create_spatial_network_map(facilities_df, h3_resolution=8):
    """Create spatial network map with H3 spatial analysis."""
    facilities_with_h3 = add_h3_indexing(facilities_df, resolution=h3_resolution)
    spatial_clusters = aggregate_by_h3_hexagon(facilities_with_h3)
    spatial_distances = calculate_spatial_distances(facilities_with_h3)

    return {
        "facilities": facilities_with_h3,
        "spatial_clusters": spatial_clusters,
        "spatial_distances": spatial_distances,
        "h3_resolution": h3_resolution,
        "spatial_analysis_method": "USEPA Sewersheds H3 Methodology",
    }


def create_folium_map(facilities_df, h3_clusters=None):
    """Create interactive Folium map showing facilities and spatial relationships."""
    center_lat = facilities_df["LATITUDE"].mean()
    center_lon = facilities_df["LONGITUDE"].mean()

    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=8, tiles="OpenStreetMap"
    )

    # Color mapping for facility types
    facility_colors = {
        "Treatment Plant": "#ADD8E6",
        "Collection: Separate Sewers": "#C4A484",
        "Collection: Pump Stations": "#C4A484",
        "Collection: Combined Sewers": "#C4A484",
        "Collection: Interceptor Sewers": "#C4A484",
        "Storage Tanks": "#808080",
        "Storage Facility": "#808080",
        "Onsite Wastewater Treatment System": "#FFD580",
        "Phase II MS4": "#FFD580",
        "Phase I MS4": "#FFD580",
        "Non-traditional MS4": "#FFD580",
        "Sanitary Landfills": "#FFD580",
        "Water Reuse": "#9370DB",
        "Resource Extraction": "#9370DB",
        "Biosolids Handling Facility": "#ADD8E6",
        "Clustered System": "#ADD8E6",
        "Other": "#FFFFC5",
    }

    # Add facility markers
    for _, facility in facilities_df.iterrows():
        if pd.notna(facility["LATITUDE"]) and pd.notna(facility["LONGITUDE"]):
            facility_type = facility.get("FACILITY_TYPE", "Other")
            color = facility_colors.get(facility_type, "#FFFFC5")

            popup_text = f"""
            <b>{facility.get('FACILITY_NAME', 'Unknown')}</b><br>
            Type: {facility_type}<br>
            CWNS ID: {facility.get('CWNS_ID', 'N/A')}<br>
            Population Served: {
                facility.get('TOTAL_RES_POPULATION_2022', 'N/A')
                }<br>
            Design Flow: {facility.get('CURRENT_DESIGN_FLOW', 'N/A')} MGD
            """

            folium.CircleMarker(
                location=[facility["LATITUDE"], facility["LONGITUDE"]],
                radius=8,
                popup=folium.Popup(popup_text, max_width=300),
                color="black",
                weight=1,
                fillColor=color,
                fillOpacity=0.7,
            ).add_to(m)

    # Add H3 cluster visualization if provided
    if h3_clusters is not None and not h3_clusters.empty:
        for _, cluster in h3_clusters.iterrows():
            if pd.notna(cluster["CENTROID_LAT"]) and pd.notna(cluster["CENTROID_LON"]):
                hex_boundary = h3.cell_to_boundary(cluster["H3_INDEX"])
                hex_coords = [[lat, lon] for lon, lat in hex_boundary]

                folium.Polygon(
                    locations=hex_coords,
                    color="red",
                    weight=2,
                    fillColor="red",
                    fillOpacity=0.1,
                    popup=f"H3 Cluster: {cluster['FACILITY_COUNT']} facilities",
                ).add_to(m)

    return m


def export_spatial_analysis(
    facilities_df, output_dir="processed_data/", h3_resolution=8
):
    """Export spatial analysis results to various formats."""
    facilities_with_h3 = add_h3_indexing(facilities_df, resolution=h3_resolution)
    spatial_clusters = aggregate_by_h3_hexagon(facilities_with_h3)
    spatial_distances = calculate_spatial_distances(facilities_with_h3)

    output_files = {}

    facilities_with_h3.to_csv(
        f"{output_dir}facilities_with_h3_indexing.csv", index=False
    )
    output_files["facilities_h3"] = f"{output_dir}facilities_with_h3_indexing.csv"

    spatial_clusters.to_csv(f"{output_dir}h3_spatial_clusters.csv", index=False)
    output_files["spatial_clusters"] = f"{output_dir}h3_spatial_clusters.csv"

    spatial_distances.to_csv(f"{output_dir}spatial_distances.csv", index=False)
    output_files["spatial_distances"] = f"{output_dir}spatial_distances.csv"

    return output_files
