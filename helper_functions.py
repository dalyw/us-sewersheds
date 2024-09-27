import pandas as pd
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt

### IMPORT DMR AND ESMR DATA

analysis_range = range(2014, 2024)
save = False
load = True

# import list of npdes codes for permits in the filtered full facilities flat file
facilities_list = pd.read_csv('data/facilities_list/NPDES+WDR Facilities List_20240906.csv')
npdes_from_facilities_list = facilities_list[facilities_list['NPDES # CA#'].notna()]['NPDES # CA#'].unique().tolist()
facility_place_id_from_facilities_list = facilities_list['FACILITY ID'].tolist()

ref_parameter = pd.read_csv('data/dmrs/REF_PARAMETER.csv')

columns_to_keep_dmr = [
    'EXTERNAL_PERMIT_NMBR',
    'LIMIT_SET_ID',
    'PARAMETER_CODE',
    'PARAMETER_DESC',
    'MONITORING_LOCATION_CODE',
    'LIMIT_VALUE_TYPE_CODE',
    'LIMIT_VALUE_NMBR',
    'LIMIT_VALUE_STANDARD_UNITS',
    'LIMIT_UNIT_CODE',
    'LIMIT_VALUE_QUALIFIER_CODE',
    'STANDARD_UNIT_CODE',
    'STATISTICAL_BASE_CODE',
    'STATISTICAL_BASE_TYPE_CODE',
    'LIMIT_FREQ_OF_ANALYSIS_CODE',
    'LIMIT_TYPE_CODE',
    'MONITORING_PERIOD_END_DATE',
    'DMR_VALUE_ID',
    'DMR_VALUE_NMBR',
    'DMR_UNIT_CODE',
    'DMR_UNIT_DESC',
    'DMR_VALUE_STANDARD_UNITS',
    'VALUE_RECEIVED_DATE',
    'NODI_CODE',
    'EXCEEDENCE_PCT'
]

columns_to_keep_esmr = [
       'parameter', 
       'qualifier', 
       'result', 
       'units', 'mdl', 'ml', 'rl',
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

def read_all_dmrs(save=False, load=True):
    if save:
        data_dict = {}
        drop_toxicity = False
        for year in analysis_range:
            data_dict[year] = read_dmr(year, drop_no_limit=True)
            # for data_dict[year], change the POLLUTANT_DESC on all rows with POLLUTANT_CODE starting with T or W into 'Toxicity'
            data_dict[year].loc[data_dict[year]['PARAMETER_CODE'].str.startswith(('T', 'W')), 'PARAMETER_DESC'] = 'Toxicity'
            if drop_toxicity:
                data_dict[year] = data_dict[year][~data_dict[year]['PARAMETER_DESC'].str.contains('Toxicity')]
    if load:
        with open('processed_data/step1/data_dict.pkl', 'rb') as f:
            data_dict = pickle.load(f)
    return data_dict

def read_limits(year):
    """
    Reads the CA DMR data for the given year
    """
    data = pd.read_csv(f'data/dmrs/CA_FY{year}_NPDES_DMRS_LIMITS/CA_FY{year}_NPDES_LIMITS.csv', low_memory=False)
    print(f'{year} limits data has {len(data)} limits and {len(data["EXTERNAL_PERMIT_NMBR"].unique())} unique permits')
    columns_to_keep = [
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
        'STATISTICAL_BASE_CODE',
        'STATISTICAL_BASE_TYPE_CODE',
        'LIMIT_VALUE_QUALIFIER_CODE',
        'LIMIT_FREQ_OF_ANALYSIS_CODE',
        'LIMIT_TYPE_CODE',
    ]
    data = data[columns_to_keep]
    return data


def read_esmr(save=False, load=True):
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
        # convert timestamp columns to datetime
        timestamp_columns = data_dict[data_dict['type'] == 'timestamp']['column']
        for col in timestamp_columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')
        data = data[columns_to_keep_esmr]
        print(f'ESMR data has {len(data)} ESMR events and {len(data["facility_place_id"].unique())} unique facilities')
        data = data[data['facility_place_id'].isin(facility_place_id_from_facilities_list)]
        print(f'ESMR data has {len(data)} ESMR events and {len(data["facility_place_id"].unique())} unique facilities that match facilities list')
        data = data[data['result'].notna()]
        print(f'ESMR data has {len(data)} ESMR events and {len(data["facility_place_id"].unique())} unique facilities that match facilities list and are not NA')
        data['sampling_date_datetime'] = pd.to_datetime(data['sampling_date'], format='%Y-%m-%d', errors='coerce')
        data = data[data['sampling_date_datetime'].dt.year.isin(analysis_range)]
        print(f'ESMR data has {len(data)} ESMR events and {len(data["facility_place_id"].unique())} unique facilities that match facilities list and are not NA and are in the analysis date range')
        data.to_csv('processed_data/esmr_data.csv', index=False)
    if load:
        data = pd.read_csv('processed_data/esmr_data.csv')
    return data


### CATEGORIZE PARAMETERS

with open('step1/parameter_sorting_dict.json', 'r') as f:
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
    # Initialize the PARENT_CATEGORY and SUB_CATEGORY columns
    df['PARENT_CATEGORY'] = 'Uncategorized'
    df['SUB_CATEGORY'] = 'Uncategorized'
    
    # Iterate through the parameter sorting dictionary
    for key, value in parameter_sorting_dict.items():
        if 'values' in value:
            mask = df[desc_column].str.contains('|'.join(value['values']), case=value.get('case', False))
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
    return desc.replace(',', '').replace('[', '(').replace(']', ')').replace(' ', '').replace("'", '').replace(".", '').replace("&", 'and').lower()

def match_parameter_desc(row, target_df):
    normalized_desc = normalize_param_desc(str(row['PARAMETER_DESC']))
    match = target_df[target_df['normalized_desc'] == normalized_desc]
    
    if match.empty:
        normalized_desc_no_sum = normalized_desc.replace(', Sum', '').replace(', Total', '').replace(', sum', '').replace(', total', '')
        match = target_df[target_df['normalized_desc'].str.replace(', Sum', '').replace(', Total', '').replace(', sum', '').replace(', total', '') == normalized_desc_no_sum]
    
    return match['PARAMETER_DESC'].iloc[0] if not match.empty else ''

