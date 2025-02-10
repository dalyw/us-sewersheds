import pandas as pd
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import re
# LIST OF FUNCTIONS:
# - read_dmr
# - read_all_dmrs
# - read_limits
# - read_esmr
# - categorize_parameters
# - plot_pie_counts
# - normalize_param_desc
# - match_parameter_desc
# - plot_facilities_map

### IMPORT DMR AND ESMR DATA
save = False
load = True

# import list of npdes codes for permits in the filtered full facilities flat file
facilities_list = pd.read_csv('data/facilities_list/NPDES+WDR Facilities List_20240906.csv')
npdes_from_facilities_list = facilities_list[facilities_list['NPDES # CA#'].notna()]['NPDES # CA#'].unique().tolist()

ref_parameter = pd.read_csv('data/dmrs/REF_PARAMETER.csv')

columns_to_keep_limits = [
        'EXTERNAL_PERMIT_NMBR',
        'LIMIT_SET_ID',
        'PARAMETER_CODE',
        'PARAMETER_DESC',
        'MONITORING_LOCATION_CODE',
        'LIMIT_VALUE_TYPE_CODE',
        'LIMIT_VALUE_NMBR',
        'LIMIT_VALUE_STANDARD_UNITS',
        'LIMIT_UNIT_CODE',
        'STANDARD_UNIT_CODE',
        'STANDARD_UNIT_DESC',
        'STATISTICAL_BASE_CODE',
        'STATISTICAL_BASE_TYPE_CODE',
        'LIMIT_VALUE_QUALIFIER_CODE',
        'LIMIT_FREQ_OF_ANALYSIS_CODE',
        'LIMIT_TYPE_CODE',
        'LIMIT_SET_DESIGNATOR',
        'LIMIT_SET_SCHEDULE_ID',
        'LIMIT_UNIT_DESC',
        'LIMIT_SAMPLE_TYPE_CODE',
    ]

columns_to_keep_dmr = columns_to_keep_limits + [
    'MONITORING_PERIOD_END_DATE',
    'DMR_VALUE_ID',
    'DMR_VALUE_NMBR',
    'DMR_UNIT_CODE',
    'DMR_UNIT_DESC',
    'DMR_VALUE_STANDARD_UNITS',
    'NODI_CODE'
]

columns_to_keep_esmr = [
       'parameter', 'qualifier', 
       'result', 'units', 'mdl', 'ml', 'rl',
       'sampling_date', 'sampling_time', 
       'review_priority_indicator', 
       'qa_codes', 'comments', 'facility_name',
       'facility_place_id', 'report_name',
       'location_desc'
]

def read_dmr(year, drop_no_limit=False):
    """
    Reads the CA DMR data for the given year
    - Keeps only the columns that are needed
    - Drops rows where the limit value is not present (if drop_no_limit is True) and where the No Data Indicator is present
    - Filters for monitoring locations 1, 2, EG, Y, or K
    - Filters for permits in the npdes_list from CIWQS flat file

    Returns the cleaned data
    """
    data = pd.read_csv(f'data/dmrs/CA_FY{year}_NPDES_DMRS_LIMITS/CA_FY{year}_NPDES_DMRS.csv', low_memory=False)
    print(f'{year} DMR data has {len(data)} DMR events and {len(data["EXTERNAL_PERMIT_NMBR"].unique())} unique permits')
    data = data[columns_to_keep_dmr] 
    if drop_no_limit:
        data = data[data['LIMIT_VALUE_NMBR'].notna()] # drop rows for monitoring without a permit limit
    data = data[data['NODI_CODE'].isna()] # drop rows where No Data Indicator is present
    data = data[data['MONITORING_LOCATION_CODE'].isin(['1', '2', 'EG', 'Y', 'K'])]
    data = data[data['EXTERNAL_PERMIT_NMBR'].isin(npdes_from_facilities_list)]
    # if data['PARAMETER_CODE'] has leading 0s, remove them
    data['PARAMETER_CODE'] = data['PARAMETER_CODE'].str.lstrip('0')
    data['POLLUTANT_CODE'] = data['PARAMETER_CODE'].map(ref_parameter.copy().set_index('PARAMETER_CODE')['POLLUTANT_CODE'])
    data['MONITORING_PERIOD_END_DATE'] = pd.to_datetime(data['MONITORING_PERIOD_END_DATE'])
    data['DMR_VALUE_STANDARD_UNITS'] = pd.to_numeric(data['DMR_VALUE_STANDARD_UNITS'], errors='coerce')
    data['MONITORING_PERIOD_END_DATE_NUMERIC'] = data['MONITORING_PERIOD_END_DATE'].dt.year + data['MONITORING_PERIOD_END_DATE'].dt.month / 12 + data['MONITORING_PERIOD_END_DATE'].dt.day / 365
    print(f'{year} DMR data has {len(data)} DMR events and {len(data["EXTERNAL_PERMIT_NMBR"].unique())} unique permits after filtering')
    return data

def read_all_dmrs(save=False, drop_toxicity=False):
    """
    Uses read_dmr to read all the CA DMR data for years in the analysis range
    - Changes the POLLUTANT_DESC on all rows with POLLUTANT_CODE starting with T or W into 'Toxicity'
    - Drops rows where the parameter description contains 'Toxicity' if drop_toxicity is True
    - Saves the data to a pickle
    """
    if save:
        data_dict = {}
        for year in analysis_range:
            data_dict[year] = read_dmr(year, drop_no_limit=True)
            # for data_dict[year], change the POLLUTANT_DESC on all rows with POLLUTANT_CODE starting with T or W into 'Toxicity'
            data_dict[year].loc[data_dict[year]['PARAMETER_CODE'].str.startswith(('T', 'W')), 'PARAMETER_DESC'] = 'Toxicity'
            if drop_toxicity:
                data_dict[year] = data_dict[year][~data_dict[year]['PARAMETER_DESC'].str.contains('Toxicity')]
        # save the data_dict to a pickle
        with open('processed_data/step3/data_dict.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
    else: # load the data_dict from the pickle
        with open('processed_data/step3/data_dict.pkl', 'rb') as f:
            data_dict = pickle.load(f)
    return data_dict

def read_limits(year):
    """
    Reads the CA DMR data for the given year
    """
    data = pd.read_csv(f'data/dmrs/CA_FY{year}_NPDES_DMRS_LIMITS/CA_FY{year}_NPDES_LIMITS.csv', low_memory=False)
    len_orig = len(data)
    data = data[data['EXTERNAL_PERMIT_NMBR'].isin(npdes_from_facilities_list)]
    print(f'{year} has {len(data)} limits and {len(data["EXTERNAL_PERMIT_NMBR"].unique())} unique permits after filtering ({len_orig} limits before filtering)')
    return data[columns_to_keep_limits]


def read_esmr(save=False):
    """
    Reads the CA ESMR data for all years since 2006
    """
    if save:
        # use data_dict to specify data types for reading the csv
        data_dict = pd.read_csv('data/esmr/esmr_data_dictionary.csv')
        dtype_dict = dict(zip(data_dict['column'], data_dict['type']))
        dtype_dict = {col: str if dtype == 'text' or dtype == 'timestamp' else float for col, dtype in dtype_dict.items()}
        # read the csv with the dtype_dict
        data = pd.read_csv('data/esmr/esmr-analytical-export_years-2006-2024_2024-09-03.csv', dtype=dtype_dict)
        fac_orig = len(data["facility_place_id"].unique())
        data = data[data['result'].notna()]
        print(f'ESMR data: {len(data)} events and {len(data["facility_place_id"].unique())} unique non-NA facilities ({fac_orig} facilities before filtering)')
        data['sampling_date_datetime'] = pd.to_datetime(data['sampling_date'], format='%Y-%m-%d', errors='coerce')
        data = data[data['sampling_date_datetime'].dt.year.isin(analysis_range)]
        print(f' --> {len(data)} events and {len(data["facility_place_id"].unique())} unique non-NA facilities ({fac_orig} facilities in year range {analysis_range[0]}-{analysis_range[-1]})')
        data[columns_to_keep_esmr].to_csv('processed_data/step1/esmr_data.csv', index=False)
    else: # load the data from the csv
        data = pd.read_csv('processed_data/step1/esmr_data.csv')
    return data[columns_to_keep_esmr]


### CATEGORIZE PARAMETERS
with open('processed_data/step1/parameter_sorting_dict.json', 'r') as f:
    parameter_sorting_dict = json.load(f)

def categorize_parameters(df, parameter_sorting_dict, desc_column):
    """
    Categorize parameters in a dataframe based on a sorting dictionary.
    
    Args:
    df (pd.DataFrame): The dataframe containing parameters to categorize.
    parameter_sorting_dict (dict): Dictionary containing categories and their associated keywords.
    desc_column (str): Name of the column or index containing parameter descriptions.
    
    Returns:
    pd.DataFrame: The input dataframe with additional 'PARENT_CATEGORY' and 'SUB_CATEGORY' columns.
    """
    df['PARENT_CATEGORY'] = 'Uncategorized'
    df['SUB_CATEGORY'] = 'Uncategorized'
    # iterate through the parameter sorting dictionary
    for key, value in parameter_sorting_dict.items():
        if 'values' in value:
            mask = df[desc_column].str.contains('|'.join(map(re.escape, value['values'])), case=value.get('case', False))
            df.loc[mask, 'PARENT_CATEGORY'] = key
            df.loc[mask, 'SUB_CATEGORY'] = key
        else:
            for sub_key, sub_value in value.items():
                mask = df[desc_column].str.contains('|'.join(sub_value['values']), case=False)
                df.loc[mask, 'PARENT_CATEGORY'] = key
                df.loc[mask, 'SUB_CATEGORY'] = sub_key
    return df

def plot_pie_counts(df, title):
        """
        Inputs: df
        Returns: none, plots figure
        """
        category_counts = df['PARENT_CATEGORY'].value_counts()
        plt.figure(figsize=(5, 5))
        plt.pie(category_counts, 
                autopct=lambda pct: f'{pct:.1f}%' if pct > 4 else '', 
                startangle=140)
        plt.title(title)
        plt.legend(category_counts.index, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.tight_layout()
        plt.show()

def normalize_param_desc(desc):
    """
    Normalize the parameter description by removing commas, brackets, spaces, apostrophes, and dots,
      converting to lowercase, and removing "sum" and "total"
    """
    return desc.replace(", sum", "").replace(", total", "").replace(", tot.", "")\
               .replace(", Sum", "").replace(", Total", "")\
               .replace(',', '').replace('[', '(').replace(']', ')').replace(' ', '')\
               .replace("'", '').replace(".", '').replace("&", 'and')\
               .lower()

def match_parameter_desc(row, target_df, target_desc_column):
    """
    Match the parameter description in the target dataframe to the parameter description in the row.
    """
    normalized_desc = normalize_param_desc(str(row['PARAMETER_DESC']))
    match = target_df[target_df['normalized_desc'] == normalized_desc]
    return match[target_desc_column].iloc[0] if not match.empty else ''


def plot_facilities_map(num_parameters_per_facility, legend_label, label_threshold):
    """
    Plot the facilities on a map of CA
    Inputs:
     - num_parameters_per_facility: dictionary with the number of parameters per facility
     - legend_label: label for the legend
     - label_threshold: threshold for the number of parameters to label the facility
     Returns: none, plots figure
    """
    ca_counties = gpd.read_file('data/ca_counties/CA_Counties.shp')
    facilities_list = pd.read_csv('data/facilities_list/NPDES+WDR Facilities List_20240906.csv')
    facilities_with_coords = facilities_list.copy()[['NPDES # CA#', 'LATITUDE DECIMAL DEGREES', 'LONGITUDE DECIMAL DEGREES']].rename(columns={'NPDES # CA#': 'NPDES_CODE', 'LATITUDE DECIMAL DEGREES': 'LATITUDE', 'LONGITUDE DECIMAL DEGREES': 'LONGITUDE'})
    facilities_with_coords_merged = pd.DataFrame({'NPDES_CODE': list(num_parameters_per_facility.keys())}).merge(facilities_with_coords, on='NPDES_CODE', how='left')
    facilities_gdf = gpd.GeoDataFrame(
        facilities_with_coords_merged,
        geometry=gpd.points_from_xy(facilities_with_coords_merged['LONGITUDE'], facilities_with_coords_merged['LATITUDE']),
        crs="EPSG:4326"
    )

    # reproject different CRS
    if facilities_gdf.crs != ca_counties.crs:
        if ca_counties.crs is None:
            ca_counties = ca_counties.set_crs(facilities_gdf.crs)
        else:
            facilities_gdf = facilities_gdf.to_crs(ca_counties.crs)
            
    fig, ax = plt.subplots(figsize=(8, 5))
    ca_counties.plot(ax=ax, color='lightgray')
    facilities_gdf['num_parameters'] = facilities_gdf['NPDES_CODE'].map(num_parameters_per_facility)
    
    # Create a colormap
    norm = plt.Normalize(vmin=facilities_gdf['num_parameters'].min(), vmax=facilities_gdf['num_parameters'].max())
    cmap = plt.cm.viridis
    
    scatter = facilities_gdf.plot(ax=ax, column='num_parameters', cmap=cmap, norm=norm, markersize=10, alpha=0.7)

    # Sort facilities by number of parameters and get top 10 with highest values
    top_facilities = facilities_gdf[facilities_gdf['num_parameters'] >= label_threshold].sort_values('num_parameters', ascending=False).head(10)

    # Create a list to store label positions
    label_positions = []
    # Sort top facilities by latitude (north to south)
    top_facilities_sorted = top_facilities.sort_values('LATITUDE', ascending=False)

    # Calculate label positions
    label_x = ax.get_xlim()[0] + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    label_y_start = ax.get_ylim()[1] - 0.57 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    label_y_step = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0])

    # Add labels and lines for top facilities
    for idx, (_, row) in enumerate(top_facilities_sorted.iterrows()):
        if not row.geometry.is_empty and pd.notna(row.geometry.x) and pd.notna(row.geometry.y):
            # Calculate label position
            label_y = label_y_start - idx * label_y_step
            
            # Add label
            ax.annotate(f"{row['NPDES_CODE']}", 
                        xy=(label_x, label_y),
                        xytext=(0, 0), 
                        textcoords="offset points",
                        fontsize=8,
                        ha='left',
                        va='center')
            
            # Add line
            ax.plot([row.geometry.x, label_x + 2.5*1e5], [row.geometry.y, label_y], 
                    color='black', linewidth=0.5, alpha=0.5)
            
            label_positions.append((label_x, label_y))
    
    # Verify sorting
    print("Facilities sorted by latitude (north to south):")
    for _, row in top_facilities_sorted.iterrows():
        print(f"NPDES_CODE: {row['NPDES_CODE']}, Latitude: {row['LATITUDE']}")

    # create a custom legend with dots for each integer
    unique_params = sorted(facilities_gdf['num_parameters'].unique())
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=str(int(value)),
                                  markerfacecolor=cmap(norm(value)), markersize=10)
                       for value in unique_params]
    ax.legend(handles=legend_elements, title=legend_label, loc='upper right', frameon=False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()