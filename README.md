# us-sewersheds

This repository includes code to visualize sewershed interconnections int he US based on the 2022 Clean Watershed Needs Survey. The us_sewersheds folder includes two scripts:

1. merge_cwns_data.py
    Merges multiple sources for population served in to the primary facilities list.

    Data used:
     -- data/cwns: Clean Watersheds Needs Survey 2022 dataset
     -- data/ww_surveillance: COVID monitoring dataset which also includes facility population served
     -- data/sso: SSO Annual Report ("Questionnaire")

2. sewersheds_app.py
    Deploys Streamlit application to visualize different sewersheds in the US, by state and county


The relevant national data is housed under "data"

The processed data, containing a pickle file used for visualizing the plots, is housed under "us_sewersheds"

The packages needed to run the scripts locally are contained in the poetry.lock file. A local environment can be created after installing Conda using the following command line prompts:
    conda create --name us-sewersheds python=3.11
    conda activate us-sewersheds
    conda install poetry
    poetry install

Then the scripts can be run;
    python us_sewersheds/merge_cwns_data.py
    streamlit run us_sewersheds/sewersheds_app.py


Notes:
Some facilities and sewersheds may be located across multiple counties. When this is the case, the sewershed is logged with its primary county, as defined by the county housing a plurality of nodes.