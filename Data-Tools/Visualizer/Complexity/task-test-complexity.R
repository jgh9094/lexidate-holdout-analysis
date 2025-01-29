# responsibe for generating all figures for holdout approach

rm(list = ls())
setwd('~/Desktop/Repositories/lexidate-variation-analysis/Data-Tools/Visualizer/Complexity')
cat("\014")

library(ggplot2)
library(cowplot)
library(dplyr)
library(PupillometryR)
library(ggpubr)
library(scales) # to access break formatting functions

NAMES = c('tournament', 'lexicase')
SHAPE <- c(21, 21)
cb_palette <- c('#DC1E34', '#004D40')
TSIZE <- 17
REPLICATES <- 30
data_dir <- './'
c_task_id_lists <- c(146818,359954,359955,190146,168757,359956,359958,359959,2073,359960,168784,359962)

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
classification_scores <- read.csv(paste(data_dir, '../../data.csv', sep = "", collapse = NULL), header = TRUE, stringsAsFactors = FALSE)
classification_scores$selection <- factor(classification_scores$selection, levels = NAMES)
classification_scores <- filter(classification_scores, task_type == 'classification')

# config for plots
col_theme = theme(legend.position = "none", axis.title.x=element_blank(),axis.title.y=element_blank(),
                  axis.ticks.x = element_blank(), axis.text.x = element_blank(), axis.text.y = element_text(angle = 90, hjust = 0.5))

test_plot <- function(task, spliter)
{
  return(filter(classification_scores, task_id == task & split == spliter) %>%
    ggplot(., aes(x = selection, y = testing_complexity, color = selection,
                  fill = selection, shape = selection)) +
    geom_flat_violin(position = position_nudge(x = 0.1, y = 0),
                     scale = "width", alpha = 0.2, width = 1.5) +
    geom_boxplot(color = "black", width = .08, outlier.shape = NA, alpha = 0.0,
                 size = 0.8, position = position_nudge(x = .15, y = 0)) +
      geom_point(position = position_jitter(width = 0.03, height = 0.0),
                 size = 1.5, alpha = 1.0) +
    scale_y_log10(
        name="Complexity",
        breaks = trans_breaks("log10", function(x) 10^x),
        labels = trans_format("log10", math_format(10^.x))
    ) +
    scale_x_discrete(
      name = "Treatment"
    ) +
    scale_shape_manual(values = SHAPE,) +
    scale_colour_manual(values = cb_palette,) +
    scale_fill_manual(values = cb_palette,) +
    ggtitle(spliter) +
    p_theme +
    guides(
      shape=guide_legend(nrow = 1, title.position = "bottom",
                         title = "Selection scheme"),
      color=guide_legend(nrow = 1, title.position = "bottom",
                         title = "Selection scheme"),
      fill=guide_legend(nrow = 1, title.position = "bottom",
                        title = "Selection scheme")))
}

test_plot_last_col <- function(task, spliter)
{
  filter(classification_scores, task_id == task & split == spliter) %>%
    ggplot(., aes(x = selection, y = testing_complexity, color = selection,
                  fill = selection, shape = selection)) +
    geom_flat_violin(position = position_nudge(x = 0.1, y = 0),
                     scale = "width", alpha = 0.2, width = 1.5) +
    geom_boxplot(color = "black", width = .08, outlier.shape = NA, alpha = 0.0,
                 size = 0.8, position = position_nudge(x = .15, y = 0)) +
    geom_point(position = position_jitter(width = .015, height = .0001),
               size = 2.0, alpha = 1.0) +
    scale_y_log10(
        name="Complexity",
        breaks = trans_breaks("log10", function(x) 10^x),
        labels = trans_format("log10", math_format(10^.x)),
        position = 'right',
        sec.axis = dup_axis()
    ) +
    scale_x_discrete(
      name = "Treatment"
    ) +
    scale_shape_manual(values = SHAPE,) +
    scale_colour_manual(values = cb_palette,) +
    scale_fill_manual(values = cb_palette,) +
    ggtitle(spliter) +
    p_theme + theme(plot.title = element_text(hjust=0.5),
                    axis.text.x.top = element_blank(), # remove ticks/text on labels
                    axis.ticks.x.top = element_blank(),
                    axis.text.y.right = element_blank(),
                    axis.ticks.y.right = element_blank(),
                    axis.title.x.bottom = element_blank(), # remove titles
                    axis.title.y.left = element_blank()) +
    guides(
      shape=guide_legend(nrow = 1, title.position = "bottom",
                         title = "Selection scheme"),
      color=guide_legend(nrow = 1, title.position = "bottom",
                         title = "Selection scheme"),
      fill=guide_legend(nrow = 1, title.position = "bottom",
                        title = "Selection scheme"))
}

test_row_p <- function(test_10, test_20, test_50, test_80, test_90)
{
  plot_grid(
    test_10 + col_theme,
    test_20 + col_theme,
    test_50 + col_theme,
    test_80 + col_theme,
    test_90 + theme(legend.position = "none", axis.title.x=element_blank(),
                     axis.ticks.x = element_blank(), axis.text.x = element_blank(),
                     axis.text.y = element_text(angle = 90, hjust = 0.5)),
    ncol = 5,
    rel_widths = c(1.0,1.0,1.0,1.0,1.0)
  )
}

source('./task-146818.R')
source('./task-359954.R')
source('./task-359955.R')
source('./task-190146.R')
source('./task-168757.R')
source('./task-359956.R')
source('./task-359958.R')
source('./task-359959.R')
source('./task-2073.R')
source('./task-359960.R')
source('./task-168784.R')
source('./task-359962.R')

c_theme <- theme(legend.position = "none", axis.title.x=element_blank(),
                         axis.ticks.x = element_blank(), axis.text.x = element_blank(),
                         axis.text.y = element_text(angle = 90, hjust = 0.5))

collection <- plot_grid(
  task_2073_plot + c_theme,
  task_146818_plot + c_theme,
  task_168757_plot + c_theme,
  task_168784_plot + c_theme,
  task_190146_plot + c_theme,
  task_359954_plot + c_theme,
  task_359955_plot + c_theme,
  task_359956_plot + c_theme,
  task_359958_plot + c_theme,
  task_359959_plot + c_theme,
  task_359960_plot + c_theme,
  task_359962_plot,
  nrow = 12,
  rel_heights = c(1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0)
)

save_plot(
  filename ="complexity.pdf",
  collection,
  base_width=12,
  base_height=30
)