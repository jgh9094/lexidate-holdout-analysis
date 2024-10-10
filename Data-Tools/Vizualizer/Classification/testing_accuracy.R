# responsibe for generating all figures for holdout approach

rm(list = ls())
setwd('~/Desktop/Repositories/lexidate-variation-analysis/Data-Tools/Vizualizer/Classification/')
cat("\014")

library(ggplot2)
library(cowplot)
library(dplyr)
library(PupillometryR)
library(ggpubr)

NAMES = c('tournament', 'lexicase')
SHAPE <- c(21, 24, 22, 25)
cb_palette <- c('#D81B60', '#1E88E5', '#FFC107', '#004D40')
TSIZE <- 17
REPLICATES <- 30
data_dir <- './'
c_task_id_lists <- c(146818, 168784, 190137, 359969)

p_theme <- theme(
  plot.title = element_text(face = "bold", size = 17, hjust=0.5),
  panel.border = element_blank(),
  panel.grid.minor = element_blank(),
  legend.title=element_text(size=17),
  legend.text=element_text(size=17),
  axis.title = element_text(size=17),
  axis.text = element_text(size=11),
  axis.text.y = element_text(angle = 90, hjust = 0.5),
  legend.position="bottom",
  panel.background = element_rect(fill = "#f1f2f5",
                                  colour = "white",
                                  size = 0.5, linetype = "solid")
)

# classification scores
classification_scores <- read.csv(paste(data_dir, '../holdout_data.csv', sep = "", collapse = NULL), header = TRUE, stringsAsFactors = FALSE)
classification_scores$selection <- factor(classification_scores$selection, levels = NAMES)
classification_scores <- filter(classification_scores, task_type == 'classification')

classification_scores <- classification_scores[duplicated(classification_scores$seed) | duplicated(classification_scores$seed, fromLast = TRUE), ] %>%
  select(testing_performance, selection, seed, task_id, split)
classification_scores <- reshape(classification_scores, timevar = 'selection', idvar = c('seed', 'task_id', 'split'), direction = "wide")

names(classification_scores)[names(classification_scores) == "testing_performance.lexicase"] <- "lexicase"
names(classification_scores)[names(classification_scores) == "testing_performance.tournament"] <- "tournament"


classification_scores$score_diff <- classification_scores$lexicase - classification_scores$tournament

# call other scripts to generate rows
source('./testing_25.R')
source('./testing_50.R')
source('./testing_75.R')

bottom = plot_grid(
  get_title(task_1),
  get_title(task_2),
  get_title(task_3),
  get_title(task_4),
  ncol=4
)

# legend
legend <- cowplot::get_legend(
  task_1 +
    guides(
      shape=guide_legend(nrow=1,title="Selection scheme"),
      color=guide_legend(nrow=1,title="Selection scheme"),
      fill=guide_legend(nrow=1,title="Selection scheme"),
    ) +
    theme(
      legend.position = "top",
      legend.box="verticle",
      legend.justification="center"
    )
)

# generate figure
testing_accuracy = plot_grid(
  ggdraw() + draw_label("Accuracy on test set per OpenML classification task", fontface='bold', size = 20) + p_theme,
  testing_25,
  testing_50,
  testing_75,
  bottom,
  legend,
  nrow=6,
  rel_heights =  c(.1, 1,1,1, 0.06, .1),
  label_size = TSIZE
)

save_plot(
  paste(filename ="testing_accuracy.pdf"),
  testing_accuracy,
  base_width=10,
  base_height=10
)
