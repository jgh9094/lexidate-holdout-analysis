# responsibe for generating all figures for holdout approach

rm(list = ls())
setwd('~/Desktop/Repositories/lexidate-variation-analysis/Data-Tools/Vizualizer')
cat("\014")

library(ggplot2)
library(cowplot)
library(dplyr)
library(PupillometryR)
library(tidyr)

NAMES = c('tournament', 'lexicase')
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
classification_scores$task_id <- factor(classification_scores$task_id)
classification_scores <- filter(classification_scores, task_type == 'classification')


classification_scores <- classification_scores[duplicated(classification_scores$seed) | duplicated(classification_scores$seed, fromLast = TRUE), ] %>%
  select(testing_performance, selection, seed, task_id, split)

classification_scores <- reshape(classification_scores, timevar = 'selection', idvar = c('seed', 'task_id', 'split'), direction = "wide")

classification_scores$score_diff <- classification_scores$testing_performance.lexicase - classification_scores$testing_performance.tournament

p25 = filter(classification_scores, split == "25%") %>%
  ggplot(., aes(x = task_id, y = score_diff)) +
  geom_flat_violin(position = position_nudge(x = 0.1, y = 0), scale = 'width', alpha = 0.2, width = 1.0, color="black", fill="black") +
  geom_boxplot(width = .08, outlier.shape = NA, alpha = 0.0, size = 0.8, position = position_nudge(x = .15, y = 0)) +
  geom_hline(yintercept=0, linetype="dashed", size=1, color="red") +
  geom_point(position = position_jitter(width = .015, height = .001), size = 2.0, alpha = 1.0, color="black", fill="black" ) +
  scale_y_continuous(
    name="lexicase-tournament",
    limits=c(-.11, .11),
    ) + ggtitle("")+coord_flip() + p_theme
  
p50 = filter(classification_scores, split == "50%") %>%
  ggplot(., aes(x = task_id, y = score_diff)) +
  geom_flat_violin(position = position_nudge(x = 0.1, y = 0), scale = 'width', alpha = 0.2, width = 1.0, color="black", fill="black") +
  geom_boxplot(width = .08, outlier.shape = NA, alpha = 0.0, size = 0.8, position = position_nudge(x = .15, y = 0)) +
  geom_hline(yintercept=0, linetype="dashed", size=1, color="red") +
  geom_point(position = position_jitter(width = .015, height = .001), size = 2.0, alpha = 1.0, color="black", fill="black" ) +
  scale_y_continuous(
    name="lexicase-tournament",
    limits=c(-.11, .11),
  ) + ggtitle("")+coord_flip() + p_theme

p75 = filter(classification_scores, split == "75%") %>%
  ggplot(., aes(x = task_id, y = score_diff)) +
  geom_flat_violin(position = position_nudge(x = 0.1, y = 0), scale = 'width', alpha = 0.2, width = 1.0, color="black", fill="black") +
  geom_boxplot(width = .08, outlier.shape = NA, alpha = 0.0, size = 0.8, position = position_nudge(x = .15, y = 0)) +
  geom_hline(yintercept=0, linetype="dashed", size=1, color="red") +
  geom_point(position = position_jitter(width = .015, height = .001), size = 2.0, alpha = 1.0, color="black", fill="black" ) +
  scale_y_continuous(
    name="lexicase-tournament",
    limits=c(-.11, .11),
  ) + ggtitle("")+coord_flip() + p_theme

grid = plot_grid(
  ggdraw() + draw_label("Difference on test set accuracy: lexicase - tournament", fontface='bold', size = 20) + p_theme,
  p25,
  p50,
  p75,
  nrow=4,
  rel_heights =  c(.1,1,1,1),
  labels = c(' ','split=25%','split=50%','split=75%'),
  label_size = 17
)

save_plot(
  paste(filename ="classification-diff.pdf"),
  grid,
  base_width=10,
  base_height=10
)
