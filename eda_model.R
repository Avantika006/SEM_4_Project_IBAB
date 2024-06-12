# Load necessary libraries
library(ggplot2)
library(tidyverse)

# Load the dataset
df <- read.csv("basic_medical_screening-2023-07-21.csv")

# Create a new column 'prediction' based on ASD and ADHD statuses
df$prediction <- ifelse(tolower(df$asd) %in% c('true', 'true.') & !is.na(df$behav_adhd), 0,
                        ifelse(tolower(df$asd) %in% c('true', 'true.'), 1,
                               ifelse(!is.na(df$behav_adhd), 2, 3)))

# Filter rows where prediction is not 0
df <- df[df$prediction != 3, ]

# Drop irrelevant columns
features <- df[, !(names(df) %in% c('subject_sp_id', 'respondent_sp_id', 'family_sf_id', 'biomother_sp_id', 'biofather_sp_id',
                                    'sex', 'current_depend_adult', 'attn_behav', 'birth_def_cns', 'birth_def_bone', 'birth_def_fac',
                                    'birth_def_gastro', 'birth_def_thorac', 'birth_def_urogen', 'dev_lang', 'gen_test',
                                    'med_cond_birth', 'med_cond_birth_def', 'med_cond_growth', 'med_cond_neuro', 'med_cond_visaud',
                                    'mood_ocd', 'prediction', 'behav_conduct', 'behav_intermitt_explos', 'behav_odd',
                                    'basic_medical_measure_validity_flag', 'birth_def_oth_calc', 'birth_oth_calc', 'etoh_subst',
                                    'gen_dx_oth_calc_self_report', 'gen_test_cgh_cma', 'gen_test_chrom_karyo', 'gen_test_ep',
                                    'gen_test_fish_angel', 'gen_test_fish_digeorge', 'gen_test_fish_williams', 'gen_test_fish_oth',
                                    'gen_test_frax', 'gen_test_id', 'gen_test_mecp2', 'gen_test_nf1', 'gen_test_noonan',
                                    'gen_test_pten', 'gen_test_tsc', 'gen_test_unknown', 'gen_test_wes', 'gen_test_wgs',
                                    'gen_test_oth_calc', 'growth_oth_calc', 'prev_study_calc', 'eval_year', 'prev_study_agre',
                                    'prev_study_asc', 'prev_study_charge', 'prev_study_earli', 'prev_study_marbles', 'prev_study_mssng',
                                    'prev_study_seed', 'prev_study_ssc', 'prev_study_vip', 'prev_study_oth_calc', 'neuro_oth_calc',
                                    'pers_dis', 'prev_study_oth_calc', 'psych_oth_calc', 'schiz', 'tics', 'asd', 'behav_adhd',
                                    'age_at_eval_months', 'age_at_eval_years', 'gen_test_aut_dd'))]

# Replace null values with 0
features[is.na(features)] <- 0

# Separate numeric and categorical columns
numeric_cols <- colnames(features)
categorical_cols <- setdiff(colnames(features), "gest_age")

# Make histograms for gestational age
ggplot(data=features, aes(x=gest_age)) +
  geom_histogram(fill="skyblue", color="black") +
  ggtitle("Histogram of Gestational Age")


# Create an empty data frame to store the counts
count_table <- data.frame(Column_Name = character(),
                          Value_1_Count = integer(),
                          Value_0_Count = integer(),
                          stringsAsFactors = FALSE)

# Loop through each column in categorical_cols
for (col in categorical_cols) {
  # Count the number of rows with value 1 and value 0 for the current column
  val_1 <- sum(features[[col]] == 1)
  val_0 <- sum(features[[col]] == 0)
  
  # Store the counts in a vector
  val <- c(val_1, val_0)
  
  # Add the counts to the count_table data frame
  count_table <- rbind(count_table, c(col, val_1, val_0))
}

# Rename the columns of the count_table data frame
colnames(count_table) <- c("Column_Name", "Value_1_Count", "Value_0_Count")

# Print the count table
print(count_table)

