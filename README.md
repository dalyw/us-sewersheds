# us-sewersheds

This repository includes code to visualize sewershed interconnections int he US based on the 2022 Clean Watershed Needs Survey. The us_sewersheds folder includes two scripts:

The facilities list of interest for the CA Wastewater Needs Assessment is housed under "data/facilities_list"


2. Population Served
    Merges multiple sources for population served in to the primary facilities list.

    Data used:
     -- data/cwns: Clean Watersheds Needs Survey 2022 dataset
     -- data/ww_surveillance: COVID monitoring dataset which also includes facility population served
     -- data/sso: SSO Annual Report ("Questionnaire")

2. sewersheds_app.py
    Deploys Streamlit application to visualize different sewersheds in the US, by state and county
