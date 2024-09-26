import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open('helper_functions/parameter_sorting_dict.json', 'r') as f:
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