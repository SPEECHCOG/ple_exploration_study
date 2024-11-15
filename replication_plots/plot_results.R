library(tidyr)
library(dplyr)
library(ggplot2)
library(grid)
library(gridExtra)
library(ggpubr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

format_csv <- function(results_file, all_ages_b, only_ple, model){
  results <- read.csv(results_file)
  
  results <- results %>%
    filter(simulation != 'sanity_check_filtered')
  
  if(only_ple){
    results <- results %>%
      filter(simulation != 'baseline')
  }
  
  results <- results %>%
    mutate(simulation=case_when(simulation == 'full_term' ~ 'Full-term',
                                simulation == 'preterm' ~ 'Preterm',
                                simulation == 'baseline' ~ 'Baseline'))
  results <- results %>%
    filter((simulation != 'Baseline') | (current_seen_hours != 0)) %>%
    filter(chronological_age < 18) %>%
    mutate(corrected_age= chronological_age) %>%
    mutate(corrected_age = case_when(
      simulation == 'Preterm' ~ corrected_age - 2.5,
      TRUE ~ corrected_age
    ))
  
  if(!all_ages_b){
    results <- results %>%
      filter(chronological_age > 6)  # checkpoints starting from 7.5 months
  }
  
  if(model=='cpc' || model=='apc'){
    results <- results %>%
      filter(model_type==model)
  }
  
  
  return(results)

}

get_plots_se <- function(results_file, test_type, x_var, y_var, plot_type, 
                         title_b, xlab_b, ylab_b, label_b,
                         all_ages_b, only_ple, model){
  results <- format_csv(results_file, all_ages_b, only_ple, model)
  
  if(test_type=='phonotactics'){ # modelling familiarity
    results <- results %>%
      mutate(d = d*-1, g = g*-1)
  }
  
  line_size = 0.6
  text_size = 18
  
  g <- ggplot(results, aes_string(x=x_var, y=y_var))
  
  if(plot_type==1 || plot_type==2 || plot_type==3){
    g <- g +
      stat_summary(fun.data=mean_se, geom='errorbar', 
                   aes(colour=simulation), 
                   )
    if(plot_type == 2 || plot_type == 3){
      g <- g +
        stat_summary(fun.data = mean_se, geom='line', 
                     aes(colour=simulation))
      if(plot_type == 3){
        g <- g +
            stat_summary(fun.data=mean_se, geom='ribbon', 
                         aes(fill=simulation), alpha=0.3)
      }
    }
  }else{
    g <- g +
      stat_summary(fun.data=mean_se, geom='smooth', aes(colour=simulation), alpha=0.3)
  }
  
  if(model=='both'){
    g <- g +
      ggh4x::facet_grid2(.~model_type, scales='free', independent='y',
                         labeller = as_labeller(c(apc='APC', cpc='CPC')))
  }

  if(title_b){
    if(test_type == 'vowel'){
      g <- g +
        ggtitle('Vowel discrimination')
    }else if(test_type == 'tone'){
      g <- g +
        ggtitle('Tone discrimination')
    }else if(test_type == 'ids'){
      g <- g +
        ggtitle('IDS preference')
    }else{
      g <- g +
        ggtitle('Phonotactics preference')
    }
  }else{
    g <- g +
      ggtitle(' ')
  }
  
  if(xlab_b){
    if(x_var == 'chronological_age'){
      g <- g +
        xlab('Simulated chronological age (months)')
    }else if(x_var == 'corrected_age'){
      g <- g +
        xlab('Simulated corrected age (months)')
    }else if(x_var == 'overall_seen_hours'){
      g <- g +
        xlab('Total speech (hours)')
    }
  }else{
    g <- g +
      xlab('')
  }
  
  
  g <- g +
    scale_shape_manual(values=c('not significant'=1, 'significant'=8)) +
    theme(text = element_text(size = 16),
          plot.title = element_text(hjust = 0.5))
  if(ylab_b){
    if(y_var == 'g'){
      g <- g +
        ylab('Effect size (g)')
    }else{
      g <- g +
        ylab('Effect size (d)')
    }
  }else{
    g <- g +
      ylab('')
  }
  
  if(label_b){
    if(test_type == 'vowel' & x_var=='chronological_age'){
      g <- g +
        theme(legend.direction ="horizontal",
              legend.background = element_rect(fill="lightgray",
                                               size=0.5, linetype="solid", 
                                               colour ="black"),
              legend.position = c(-0.1, -0.5))
    }else if(test_type == 'tone' || (test_type == 'vowel' & x_var=='corrected_age')){
      g <- g +
        theme(legend.direction ="horizontal",
              legend.background = element_rect(fill="lightgray",
                                               size=0.5, linetype="solid", 
                                               colour ="black"),
              legend.position = c(0.5, -0.5))
    }
  }else{
    g <- g +
      theme(legend.position='none')
  }
  
  g <- g +
    scale_colour_manual(values=c('Baseline'= '#337538', 'Full-term'='#2e2585', 'Preterm'='#7e2954')) +
    scale_fill_manual(values=c('Baseline'= '#337538', 'Full-term'='#2e2585', 'Preterm'='#7e2954'))
  
  return(g)
}

# Chronological
g1_se <- get_plots_se('./results/vowel_discrimination_results_months.csv', 'vowel', 'chronological_age', 'd', 3, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, 'both')
g2_se <- get_plots_se('./results/ids_results_months.csv', 'ids', 'chronological_age', 'd', 3, TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, 'both')
g3_se <- get_plots_se('./results/phonotactics_preference_results_months.csv', 'phonotactics', 'chronological_age', 'd', 3, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, 'both')
g4_se <- get_plots_se('./results/tone_discrimination_results_months.csv', 'tone', 'chronological_age', 'd', 3, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, 'both')

# Corrected
g5_se <- get_plots_se('./results/vowel_discrimination_results_months.csv', 'vowel', 'corrected_age', 'd', 3, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, 'both')
g6_se <- get_plots_se('./results/ids_results_months.csv', 'ids', 'corrected_age', 'd', 3, TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, 'both')
g7_se <- get_plots_se('./results/phonotactics_preference_results_months.csv', 'phonotactics', 'corrected_age', 'd', 3, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, 'both')
g8_se <- get_plots_se('./results/tone_discrimination_results_months.csv', 'tone', 'corrected_age', 'd', 3, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, 'both')

# Speech
g9_se <- get_plots_se('./results/vowel_discrimination_results_months.csv', 'vowel', 'overall_seen_hours', 'd', 3, TRUE, TRUE, FALSE, TRUE, TRUE, FALSE, 'both')
g10_se <- get_plots_se('./results/ids_results_months.csv', 'ids', 'overall_seen_hours', 'd', 3, TRUE, TRUE, TRUE, FALSE, TRUE, FALSE, 'both')
g11_se <- get_plots_se('./results/phonotactics_preference_results_months.csv', 'phonotactics', 'overall_seen_hours', 'd', 3, TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, 'both')
g12_se <- get_plots_se('./results/tone_discrimination_results_months.csv', 'tone', 'overall_seen_hours', 'd', 3, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, 'both')

title <- textGrob(expression(bold('Developmental trajectories')), 
                  gp=gpar(fontsize=20))
empty <- textGrob(expression(' '), gp=gpar(fontsize=20))
layout <- rbind(c(1,1,1,1), 
                c(2,2,3,3), 
                c(2,2,3,3), 
                c(2,2,3,3), 
                c(2,2,3,3),
                c(4,4,5,5), 
                c(4,4,5,5),
                c(4,4,5,5), 
                c(4,4,5,5),
                c(6,6,6,6))

plot_type = 'chronological'  # speech, chronological, corrected

if(plot_type=='chronological'){
  grid.arrange(title, g3_se, g4_se, g2_se, g1_se, empty, layout_matrix=layout)
} else if (plot_type == 'corrected'){
  grid.arrange(title, g7_se, g8_se, g6_se, g5_se, layout_matrix=layout)
} else{
  grid.arrange(title, g11_se, g12_se, g10_se, g9_se, layout_matrix=layout)
}

# All views for one model and two capabilities

model_plot = 'cpc'
all_ages_plot = TRUE

d1_se <- get_plots_se('./results/phonotactics_preference_results_months.csv', 'phonotactics', 'chronological_age', 'd', 3, FALSE, FALSE, TRUE, FALSE, all_ages_plot, TRUE, model_plot)
d2_se <- get_plots_se('./results/tone_discrimination_results_months.csv', 'tone', 'chronological_age', 'd', 3, FALSE, TRUE, TRUE, FALSE, all_ages_plot, TRUE, model_plot)

d3_se <- get_plots_se('./results/phonotactics_preference_results_months.csv', 'phonotactics', 'corrected_age', 'd', 3, TRUE, FALSE, FALSE, FALSE, all_ages_plot, TRUE, model_plot)
d4_se <- get_plots_se('./results/tone_discrimination_results_months.csv', 'tone', 'corrected_age', 'd', 3, TRUE, TRUE, FALSE, TRUE, all_ages_plot, TRUE, model_plot)

d5_se <- get_plots_se('./results/phonotactics_preference_results_months.csv', 'phonotactics', 'overall_seen_hours', 'd', 3, FALSE, FALSE, FALSE, FALSE, all_ages_plot, TRUE, model_plot)
d6_se <- get_plots_se('./results/tone_discrimination_results_months.csv', 'tone', 'overall_seen_hours', 'd', 3, FALSE, TRUE, FALSE, FALSE, all_ages_plot, TRUE, model_plot)

title2 <- textGrob(expression(bold('Developmental trajectories (CPC Model)')), 
                  gp=gpar(fontsize=20))
layout_d <- rbind(c(1,1,1,1,1,1), 
                  c(2,2,3,3,4,4),
                  c(2,2,3,3,4,4),
                  c(2,2,3,3,4,4),
                  c(2,2,3,3,4,4),
                  c(5,5,6,6,7,7),
                  c(5,5,6,6,7,7),
                  c(5,5,6,6,7,7),
                  c(5,5,6,6,7,7),
                  c(8,8,8,8,8,8))
grid.arrange(title2, d1_se, d3_se, d5_se, d2_se, d4_se, d6_se, empty, layout_matrix=layout_d)


# Appendix plots
# CPC: IDS and Vowel discrimination

d7_se <- get_plots_se('./results/ids_results_months.csv', 'ids', 'chronological_age', 'd', 3, FALSE, FALSE, TRUE, FALSE, all_ages_plot, TRUE, model_plot)
d8_se <- get_plots_se('./results/vowel_discrimination_results_months.csv', 'vowel', 'chronological_age', 'd', 3, FALSE, TRUE, TRUE, FALSE, all_ages_plot, TRUE, model_plot)

d9_se <- get_plots_se('./results/ids_results_months.csv', 'ids', 'corrected_age', 'd', 3, TRUE, FALSE, FALSE, FALSE, all_ages_plot, TRUE, model_plot)
d10_se <- get_plots_se('./results/vowel_discrimination_results_months.csv', 'vowel', 'corrected_age', 'd', 3, TRUE, TRUE, FALSE, TRUE, all_ages_plot, TRUE, model_plot)

d11_se <- get_plots_se('./results/ids_results_months.csv', 'ids', 'overall_seen_hours', 'd', 3, FALSE, FALSE, FALSE, FALSE, all_ages_plot, TRUE, model_plot)
d12_se <- get_plots_se('./results/vowel_discrimination_results_months.csv', 'vowel', 'overall_seen_hours', 'd', 3, FALSE, TRUE, FALSE, FALSE, all_ages_plot, TRUE, model_plot)

grid.arrange(title2, d7_se, d9_se, d11_se, d8_se, d10_se, d12_se, empty, layout_matrix=layout_d)

# APC: Phonotactics and Tone discrimination
model_plot2 = 'apc'
d13_se <- get_plots_se('./results/phonotactics_preference_results_months.csv', 'phonotactics', 'chronological_age', 'd', 3, FALSE, FALSE, TRUE, FALSE, all_ages_plot, TRUE, model_plot2)
d14_se <- get_plots_se('./results/tone_discrimination_results_months.csv', 'tone', 'chronological_age', 'd', 3, FALSE, TRUE, TRUE, FALSE, all_ages_plot, TRUE, model_plot2)

d15_se <- get_plots_se('./results/phonotactics_preference_results_months.csv', 'phonotactics', 'corrected_age', 'd', 3, TRUE, FALSE, FALSE, FALSE, all_ages_plot, TRUE, model_plot2)
d16_se <- get_plots_se('./results/tone_discrimination_results_months.csv', 'tone', 'corrected_age', 'd', 3, TRUE, TRUE, FALSE, TRUE, all_ages_plot, TRUE, model_plot2)

d17_se <- get_plots_se('./results/phonotactics_preference_results_months.csv', 'phonotactics', 'overall_seen_hours', 'd', 3, FALSE, FALSE, FALSE, FALSE, all_ages_plot, TRUE, model_plot2)
d18_se <- get_plots_se('./results/tone_discrimination_results_months.csv', 'tone', 'overall_seen_hours', 'd', 3, FALSE, TRUE, FALSE, FALSE, all_ages_plot, TRUE, model_plot2)

title3 <- textGrob(expression(bold('Developmental trajectories (APC Model)')), 
                   gp=gpar(fontsize=20))

grid.arrange(title3, d13_se, d15_se, d17_se, d14_se, d16_se, d18_se, empty, layout_matrix=layout_d)

# APC: IDS and Vowel discrimination

d19_se <- get_plots_se('./results/ids_results_months.csv', 'ids', 'chronological_age', 'd', 3, FALSE, FALSE, TRUE, FALSE, all_ages_plot, TRUE, model_plot2)
d20_se <- get_plots_se('./results/vowel_discrimination_results_months.csv', 'vowel', 'chronological_age', 'd', 3, FALSE, TRUE, TRUE, FALSE, all_ages_plot, TRUE, model_plot2)

d21_se <- get_plots_se('./results/ids_results_months.csv', 'ids', 'corrected_age', 'd', 3, TRUE, FALSE, FALSE, FALSE, all_ages_plot, TRUE, model_plot2)
d22_se <- get_plots_se('./results/vowel_discrimination_results_months.csv', 'vowel', 'corrected_age', 'd', 3, TRUE, TRUE, FALSE, TRUE, all_ages_plot, TRUE, model_plot2)

d23_se <- get_plots_se('./results/ids_results_months.csv', 'ids', 'overall_seen_hours', 'd', 3, FALSE, FALSE, FALSE, FALSE, all_ages_plot, TRUE, model_plot2)
d24_se <- get_plots_se('./results/vowel_discrimination_results_months.csv', 'vowel', 'overall_seen_hours', 'd', 3, FALSE, TRUE, FALSE, FALSE, all_ages_plot, TRUE, model_plot2)

grid.arrange(title3, d19_se, d21_se, d23_se, d20_se, d22_se, d24_se, empty, layout_matrix=layout_d)




