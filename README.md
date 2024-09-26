# ca-sewersheds

This repository includes code to analyze and visualize risk varriables associated with California wastewater treatment plants. The top-level folder includes five files which can be run in order:

The facilities list of interest for the CA Wastewater Needs Assessment is housed under "data/facilities_list"

The .ipynb scripts do the following
1. Parameter Categorization
    Standardizes parameter names from different data sources, for use in steps 3 and 4.

    Data used:
     -- data/dmrs: EPA ICIS DMR datasets
     -- data/ir: California Integrated Report 303d list
     -- data/esmr: analytical results from electronic self-mointoring reporst (eSMRs) from CIWQs database

2. Population Served
    Merges multiple sources for population served in to the primary facilities list.

    Data used:
     -- data/cwns: Clean Watersheds Needs Survey 2022 dataset
     -- data/ww_surveillance: COVID monitoring dataset which also includes facility population served
     -- data/sso: SSO Annual Report ("Questionnaire")

3. Near Exceedence
    Analyzes historical effluent dta to determine which facilities are frequently at or near their permitted limits for various parameters.

    Data used:
     -- data/dmrs: EPA ICIS DMR datasets
     -- data/esmr: analytical results from electronic self-mointoring reporst (eSMRs) from CIWQs database

4. Future Limits (Proximity to Impaired Waters)
    Assesses which facilities discharge into newly-listed impaired water bodies but do not yet have a permitted limit for the listed parameters.

    Data used:
     -- data/ir: California Integrated Report 303d list

5. Generate updated facilities list
    Uses outputs from steps 2, 3 and 4 to generated an updated facilities list that gives the results of analysis
