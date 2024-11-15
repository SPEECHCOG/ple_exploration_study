# This script processes the meta-analyses data to build comparative matrices
# 1) is there evidence of an effect for age range X?
# 2) how many effects were used for the point estimate?

library(tidyverse)
library(lme4)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd('../ids_preference/')
source('./replication_analyses.R')
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd('../vowel_discrimination/')
source('./replication_analyses.R')
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


# Checkpoints in months
months <- c(7.5,9,10.5,12)
checkpoints = data.frame(age_mo = months)

# IDS preference
ids_lm = lm(d_z~age_mo, data = ds_zt_nae)

ids_checkpoints = predict(ids_lm, checkpoints, interval = 'confidence')

# Vowel discrimination 
# age not significant, effect size = mean effect for all age ranges
vowel_mo = 30.42
vowelnat_checkpoints = matrix(
  rep(c(mean_es_nat, mean_es_nat_ci.lb, mean_es_nat_ci.ub),
      each=nrow(checkpoints)), 
  nrow=nrow(checkpoints))

# Number of effects used to calculate the point estimate
# centred within 3-mo range

# IDS preference & Vowel discrimination
ids_final = c()
vnat_final = c()

df_lower_bound = data.frame(matrix(ncol=length(months)+1, nrow = 0))

column_names = c('Capability')
for (stage in months){
  column_names = c(column_names, as.character(stage))
}

colnames(df_lower_bound) <- column_names


for(i in 1:length(months)){
  checkpoint = months[i]
  # IDS preference
  ids_age_range = ds_zt_nae %>% 
    group_by() %>% 
    summarise(min=min(age_mo), max=max(age_mo))
  
  df_lower_bound[1,1] = 'IDS Preference'
  
  if(ids_age_range['min']> checkpoint || ids_age_range['max']<checkpoint){
    ids_final = c(ids_final, 'N/A') # Out of age range of the meta-analysis
    df_lower_bound[1,i+1] = 'N/A'
  }else if(sign(ids_checkpoints[i,2]) != sign(ids_checkpoints[i,3])){
    ids_final = c(ids_final, 'n.s.') # No significant effect
    df_lower_bound[1, i+1] = 'n.s.'
  }else{
    ids_final = c(ids_final, ids_checkpoints[i,1])
    if(ids_checkpoints[i,1]<0){
      df_lower_bound[1,i+1] = ids_checkpoints[i,3]  
    }else{
      df_lower_bound[1,i+1] = ids_checkpoints[i,2]  
    }
  }
  
  # Vowel disc. nat.
  vowel_nat_age_range = nat_vowels %>% 
    group_by() %>%
    summarise(min=min(mean_age_1/vowel_mo), max=max(mean_age_1/vowel_mo))
  
  df_lower_bound[2,1] = 'Vowel discrimination'
  
  if(vowel_nat_age_range['min']>checkpoint || 
     vowel_nat_age_range['max']<checkpoint){
    vnat_final = c(vnat_final, 'N/A')
    df_lower_bound[2, i+1] = 'N/A'
  }else if(sign(vowelnat_checkpoints[i,2]) != sign(vowelnat_checkpoints[i,3])){
    vnat_final = c(vnat_final, 'n.s.') 
    df_lower_bound[2, i+1] = 'n.s.'
  }else{
    vnat_final = c(vnat_final, vowelnat_checkpoints[i,1])
    if(vowelnat_checkpoints[i,1]<0){
      df_lower_bound[2, i+1] = vowelnat_checkpoints[i,3]  
    }else{
      df_lower_bound[2, i+1] = vowelnat_checkpoints[i,2]
    }
    
  }
}

ids_final = c('IDS Preference', ids_final)
vnat_final = c('Vowel discrimination', vnat_final)
final_matrix = rbind(ids_final, vnat_final)

# Formatting
colnames(final_matrix) <- column_names

write.csv(df_lower_bound, './summary_table_lb.csv', row.names = FALSE)
write.csv(final_matrix, './summary_table_es.csv', row.names = FALSE)

# Summary from the binary results of tone discrimination and phonotactics preference
tone_discrimination = read.csv('../tone_discrimination/human_data.csv')
colnames(tone_discrimination) <- c('infant_type','7.5','9','10.5','12')
phonotactics_preference = read.csv('../phonotactics_preference/human_data.csv')
colnames(phonotactics_preference) <- c('infant_type','7.5','9','10.5','12')
tone_discrimination <- tone_discrimination %>%
  mutate(Capability='Tone discriminatoin') %>%
  select(Capability, infant_type, '7.5', '9', '10.5', '12')
phonotactics_preference <- phonotactics_preference %>%
  mutate(Capability='Phonotactics preference') %>%
  select(Capability, infant_type, '7.5', '9', '10.5', '12')
ple_reference = rbind(tone_discrimination, phonotactics_preference)
write.csv(ple_reference, './summary_ple.csv', row.names = FALSE)

ple_reference <- ple_reference %>%
  mutate(Capability= paste(Capability, paste0('(', infant_type, ')'), sep=' ')) %>%
  select(-infant_type)

write.csv(rbind(df_lower_bound, ple_reference), './summary_reference_data_lb.csv', row.names = FALSE)
write.csv(rbind(final_matrix, ple_reference), './summary_reference_data_es.csv', row.names = FALSE)

# Get simulation results
ids_simulations <- read.csv('../../results/ids_results_months.csv')  # use d
vowel_simulations <- read.csv('../../results/vowel_discrimination_results_months.csv')  # use g
tone_simulations <- read.csv('../../results/tone_discrimination_results_months.csv')  # use d
phonotactics_simulations <- read.csv('../../results/phonotactics_preference_results_months.csv')  # use d

# Simulations: full-term, preterm, baseline
tests <- rbind(ids_simulations, tone_simulations, phonotactics_simulations)

# calculate mean es across runs
tests <- tests %>%
  filter(simulation %in% c('full_term', 'preterm', 'baseline')) %>%
  select(type, d, model_type, simulation, run, chronological_age) %>%
  group_by(type, model_type, simulation, chronological_age) %>%
  summarise(es = mean(d), se=sd(d)/sqrt(n())) %>%
  ungroup() %>%
  mutate(significant=ifelse(es-se<0 & 0<es+se, 'not significant', 'significant'))  # Significant at p<=0.05
  

vowel_simulations <- vowel_simulations %>%
  filter(simulation %in% c('full_term', 'preterm', 'baseline')) %>%
  select(type, g, model_type, simulation, run, chronological_age) %>%
  group_by(type, model_type, simulation, chronological_age) %>%
  summarise(es = mean(g), se=sd(g)/sqrt(n())) %>%
  ungroup() %>%
  mutate(significant=ifelse(es-se<0 & 0<es+se, 'not significant', 'significant'))  # Significant at p<=0.05


tests <- rbind(tests, vowel_simulations) %>%
  filter(chronological_age>=7.5, chronological_age<=12) %>%
  mutate(Capability=case_when(type=='ids_preference' ~ 'IDS preference',
                              type=='tone_discrimination' ~ 'Tone discrimination',
                              type=='phonotactics_preference' ~ 'Phonotactics preference',
                              type=='native' ~ 'Vowel discrimination')) %>%
  select(es, significant, model_type, simulation, chronological_age, Capability)

write.csv(tests, './summary_simulations.csv', row.names = FALSE)  
