# converted from .ipynb files using Claude 3.5 Sonnet

# Load required libraries
library(tidyverse)
library(readr)

source('helper_functions.R')

# Import Parameter Code Lists
ref_parameter_df <- read_csv('data/dmrs/REF_Parameter.csv')
ir_parameter_df <- read_csv('data/ir/ir_parameter_list.csv')
esmr_parameter_df <- read_csv('data/esmr/esmr_parameters.csv')

ir_parameter_df <- ir_parameter_df %>% select(-CATEGORY)
esmr_parameter_df <- esmr_parameter_df %>% select(-CATEGORY)

# Find Parameters with Limits in CA based on DMR Dataset
data_dict <- read_all_dmrs(save = TRUE, load = FALSE)

all_pollutants <- c()
all_parameter_codes <- c()
# Debug: Print the contents of data_dict and analysis_range
print("Contents of data_dict:")
print(names(data_dict))
print("analysis_range:")
print(analysis_range)

for (year in analysis_range) {
  if (!is.null(data_dict[[as.character(year)]])) {
    all_pollutants <- c(all_pollutants, unique(data_dict[[as.character(year)]]$POLLUTANT_CODE))
    all_parameter_codes <- c(all_parameter_codes, unique(data_dict[[as.character(year)]]$PARAMETER_CODE))
  } else {
    warning(paste("No data found for year:", year))
  }
}

# Debug: Print the results
print("Unique pollutants found:")
print(length(unique(all_pollutants)))
print("Unique parameter codes found:")
print(length(unique(all_parameter_codes)))
unique_pollutants <- unique(all_pollutants)
unique_parameter_codes <- unique(all_parameter_codes)

cat(sprintf("%d unique pollutants and %d unique parameters\n", 
            length(unique_pollutants), length(unique_parameter_codes)))

parameter_sorting_dict <- jsonlite::fromJSON('processed_data/step1/parameter_sorting_dict.json')
ref_parameter_df <- categorize_parameters(ref_parameter_df, parameter_sorting_dict, 'PARAMETER_DESC')
ir_parameter_df <- categorize_parameters(ir_parameter_df, parameter_sorting_dict, 'IR_PARAMETER_DESC')
esmr_parameter_df <- categorize_parameters(esmr_parameter_df, parameter_sorting_dict, 'ESMR_PARAMETER_DESC')

# Additional categorization of Total Toxics
ref_parameter_df <- ref_parameter_df %>%
  mutate(PARENT_CATEGORY = ifelse(str_starts(PARAMETER_CODE, "T") | str_starts(PARAMETER_CODE, "W"), 
                                  "Total Toxics", PARENT_CATEGORY),
         SUB_CATEGORY = ifelse(str_starts(PARAMETER_CODE, "T") | str_starts(PARAMETER_CODE, "W"), 
                               "", SUB_CATEGORY))

# Save to csv
write_csv(ref_parameter_df, 'processed_data/step1/REF_Parameter_categorized.csv')

plot_pie_counts <- function(df, title) {
  # Inputs: df
  # Returns: none, plots figure
  
  category_counts <- table(df$PARENT_CATEGORY)
  
  pie(category_counts, 
      labels = ifelse(category_counts/sum(category_counts) > 0.04, 
                      paste0(round(100 * category_counts/sum(category_counts), 1), "%"), 
                      ""),
      main = title,
      col = rainbow(length(category_counts)),
      init.angle = 140)
  
  legend("topright", 
         legend = names(category_counts), 
         fill = rainbow(length(category_counts)),
         cex = 0.8,
         bty = "n")
}
        
plot_pie_counts(ref_parameter_df, 'REF_Parameter Categories')
plot_pie_counts(ir_parameter_df, 'ir_parameter_df Categories')
plot_pie_counts(esmr_parameter_df, 'esmr_parameter_df Categories')


### Create list of parameter codes, names, and matched ESMR parameter names

dmr_params_df <- data.frame(PARAMETER_CODE = unique_parameter_codes) %>%
  left_join(
    ref_parameter_df %>%
      select(PARAMETER_CODE, PARAMETER_DESC, POLLUTANT_CODE, PARENT_CATEGORY, SUB_CATEGORY) %>%
      mutate(POLLUTANT_CODE = as.integer(POLLUTANT_CODE)),
    by = "PARAMETER_CODE"
  )

if (save) {
  esmr_data <- read_esmr(save = FALSE, load = TRUE)
  esmr_params_df <- data.frame(PARAMETER_DESC = unique(esmr_data$parameter)) %>%
    left_join(
      dmr_params_df %>% select(PARAMETER_DESC, PARAMETER_CODE),
      by = "PARAMETER_DESC"
    )
  write_csv(esmr_params_df, 'processed_data/step1/esmr_unique_parameters.csv')
}
if (load) {
  esmr_params_df <- read_csv('processed_data/step1/esmr_unique_parameters.csv')
}

esmr_params_df <- esmr_params_df %>%
  mutate(normalized_desc = sapply(PARAMETER_DESC, normalize_param_desc))

dmr_params_df <- dmr_params_df %>%
  rowwise() %>%
  mutate(ESMR_PARAMETER_DESC_MATCHED = match_parameter_desc(., esmr_params_df)) %>%
  ungroup()

cat(sprintf("%d out of %d parameter names automatically matched to ESMR PARAMETER_DESC in REF_PARAMETER.csv\n",
            length(unique(dmr_params_df$ESMR_PARAMETER_DESC_MATCHED)) - 1, nrow(dmr_params_df)))

dmr_esmr_mapping_manual <- read_csv('processed_data/step1/dmr_esmr_mapping_manual.csv')
dmr_params_df <- dmr_params_df %>%
  left_join(dmr_esmr_mapping_manual %>% select(PARAMETER_CODE, ESMR_PARAMETER_DESC_MANUAL),
            by = "PARAMETER_CODE") %>%
  mutate(ESMR_PARAMETER_DESC_MANUAL = replace_na(ESMR_PARAMETER_DESC_MANUAL, ""))

cat(sprintf("%d out of %d parameter names manually mapped to ESMR PARAMETER_DESC in REF_PARAMETER.csv\n",
            length(unique(dmr_params_df$ESMR_PARAMETER_DESC_MANUAL)) - 1, nrow(dmr_params_df)))

dmr_params_df <- dmr_params_df %>%
  mutate(ESMR_PARAMETER_DESC = case_when(
    ESMR_PARAMETER_DESC_MATCHED != "" ~ ESMR_PARAMETER_DESC_MATCHED,
    ESMR_PARAMETER_DESC_MANUAL != "" ~ ESMR_PARAMETER_DESC_MANUAL,
    TRUE ~ ""
  )) %>%
  select(-ESMR_PARAMETER_DESC_MATCHED, -ESMR_PARAMETER_DESC_MANUAL) %>%
  rename(DMR_PARAMETER_DESC = PARAMETER_DESC)

write_csv(dmr_params_df, 'processed_data/step1/dmr_esmr_mapping.csv')