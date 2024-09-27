# converted from .ipynb files using Claude 3.5 Sonnet

# Load required libraries
library(dplyr)
library(readr)
library(tidyr)
library(ggplot2)
library(sf)

# Helper function to read limits (assuming it's similar to the Python version)
read_limits <- function(year) {
  # Implement the read_limits function based on your specific requirements
}

# Read and process data
dmr_esmr_ir_mapping <- read_csv('processed_data/step1/dmr_esmr_ir_mapping.csv')
parent_category_map <- setNames(dmr_esmr_ir_mapping$PARENT_CATEGORY, dmr_esmr_ir_mapping$PARAMETER_CODE)
sub_category_map <- setNames(dmr_esmr_ir_mapping$SUB_CATEGORY, dmr_esmr_ir_mapping$PARAMETER_CODE)

categories <- c('Temperature', 'Metals', 'Dissolved Solids', 'Nitrogen', 'Phosphorus', 
                'Disinfectants', 'Dissolved Oxygen', 'Pathogens', 'Toxic Inorganics',
                'Turbidity', 'Color')

facilities_list <- read_csv('data/facilities_list/NPDES+WDR Facilities List_20240906.csv')

limits_2023 <- read_limits(2023) %>%
  mutate(
    PARENT_CATEGORY = parent_category_map[PARAMETER_CODE],
    SUB_CATEGORY = sub_category_map[PARAMETER_CODE]
  )

# Import and process 303d lists for 2018 and 2024
columns_to_keep <- c('Water Body CALWNUMS', 'Pollutant', 'Pollutant Category', 'Decision Status', 
                     'TMDL Requirement Status', 'Sources', 'Expected TMDL Completion Date', 
                     'Expected Attainment Date')

impaired_303d_2018 <- read_csv('data/ir/2018-303d.csv', skip = 2) %>%
  select(all_of(columns_to_keep)) %>%
  filter(!is.na(`Water Body CALWNUMS`))

impaired_303d_2024 <- read_csv('data/ir/2024-303d.csv', skip = 1) %>%
  select(all_of(columns_to_keep)) %>%
  filter(!is.na(`Water Body CALWNUMS`))

# Create lists to store the newly impaired water bodies for each category
newly_impaired_water_bodies <- list()
impaired_water_bodies <- list()

for (category in categories) {
  impaired_set_2018 <- impaired_303d_2018 %>%
    filter(`Pollutant Category` == category) %>%
    pull(`Water Body CALWNUMS`) %>%
    unique()
  
  impaired_set_2024 <- impaired_303d_2024 %>%
    filter(`Pollutant Category` == category) %>%
    pull(`Water Body CALWNUMS`) %>%
    unique()
  
  newly_impaired_water_bodies[[category]] <- setdiff(impaired_set_2024, impaired_set_2018)
  impaired_water_bodies[[category]] <- impaired_set_2024
}

# Helper function to check if watershed contains impaired water body
contains_impaired_water_body <- function(watershed_name, impaired_water_bodies) {
  if (is.na(watershed_name)) return(FALSE)
  any(sapply(impaired_water_bodies, function(x) grepl(x, watershed_name)))
}

# Check for facilities discharging to newly impaired water bodies
for (category in categories) {
  print(category)
  
  facilities_list[[paste0('Discharges to Newly ', category, ' Impaired')]] <- 
    sapply(facilities_list$`CAL WATERSHED NAME`, 
           contains_impaired_water_body, 
           impaired_water_bodies = newly_impaired_water_bodies[[category]])
  
  facilities_list[[paste0('Discharges to ', category, ' Impaired')]] <- 
    sapply(facilities_list$`CAL WATERSHED NAME`, 
           contains_impaired_water_body, 
           impaired_water_bodies = impaired_water_bodies[[category]])
  
  facilities_list[[paste0('Discharges to Newly ', category, ' Impaired and Not Limited')]] <- FALSE
  
  for (i in 1:nrow(facilities_list)) {
    sub_limits_2023 <- limits_2023 %>% 
      filter(EXTERNAL_PERMIT_NMBR == facilities_list$`NPDES # CA#`[i])
    
    if (facilities_list[[paste0('Discharges to Newly ', category, ' Impaired')]][i]) {
      for (j in 1:nrow(sub_limits_2023)) {
        if (sub_limits_2023$SUB_CATEGORY[j] == category && 
            (is.na(sub_limits_2023$LIMIT_VALUE_NMBR[j]) || 
             sub_limits_2023$LIMIT_VALUE_NMBR[j] == '')) {
          facilities_list[[paste0('Discharges to Newly ', category, ' Impaired and Not Limited')]][i] <- TRUE
          break
        }
      }
      if (!facilities_list[[paste0('Discharges to Newly ', category, ' Impaired and Not Limited')]][i]) {
        facilities_list[[paste0('Discharges to Newly ', category, ' Impaired and Not Limited')]][i] <- TRUE
      }
    }
  }
}

# Consolidate to a single column
facilities_list <- facilities_list %>%
  mutate(`Discharges to Impaired Water Bodies and Not Limited` = 
           apply(select(., starts_with("Discharges to Newly") & ends_with("Impaired and Not Limited")), 1, 
                 function(x) paste(names(x)[x], collapse = " and ")))

# Save the processed data
write_csv(facilities_list, 'processed_data/facilities_with_future_limits.csv')

# Visualization code would need to be adapted for R, potentially using ggplot2
# The mapping part would require the sf package for spatial data handling in R