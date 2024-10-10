# split == 75
# regression

# plot for task 359934
task_1 = filter(regression_scores, task_id == r_task_id_lists[1] & split == "75%") %>%
  ggpaired(.,
           color = 'black',
           cond1 = "tournament",
           cond2 = "lexicase",
           fill = "condition",
           palette = "rickandmorty",
           point.size = 2.0,
           width = 0.1,
           ggtheme = theme_gray()) + p_theme +
  scale_y_log10(
    name="Average abolute error",
    breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x))
  ) +
  scale_x_discrete(
    name="Treatment"
  ) + ggtitle('Task 359934')

# plot for task 359945
task_2 = filter(regression_scores, task_id == r_task_id_lists[2] & split == "75%") %>%
  ggpaired(.,
           color = 'black',
           cond1 = "tournament",
           cond2 = "lexicase",
           fill = "condition",
           palette = "rickandmorty",
           point.size = 2.0,
           width = 0.1,
           ggtheme = theme_gray()) + p_theme +
  scale_y_log10(
    name="Average abolute error",
    breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x))
  ) +
  scale_x_discrete(
    name="Treatment"
  ) + ggtitle('Task 359945')

# plot for task 359948
task_3 = filter(regression_scores, task_id == r_task_id_lists[3] & split == "75%") %>%
  ggpaired(.,
           color = 'black',
           cond1 = "tournament",
           cond2 = "lexicase",
           fill = "condition",
           palette = "rickandmorty",
           point.size = 2.0,
           width = 0.1,
           ggtheme = theme_gray()) + p_theme +
  scale_y_log10(
    name="Average abolute error",
    breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x))
  ) +
  scale_x_discrete(
    name="Treatment"
  ) + ggtitle('Task 359948')

# plot for task 359933
task_4 = filter(regression_scores, task_id == r_task_id_lists[4] & split == "75%") %>%
  ggpaired(.,
           color = 'black',
           cond1 = "tournament",
           cond2 = "lexicase",
           fill = "condition",
           palette = "rickandmorty",
           point.size = 2.0,
           width = 0.1,
           ggtheme = theme_gray()) + p_theme +
  scale_y_log10(
    breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x)),
    name = "Split of 75%",
    sec.axis = dup_axis()
  ) +
  scale_x_discrete(
    name="Treatment"
  )+ ggtitle('Task 359933')
  ggtitle("")+
  p_theme +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.x.top = element_blank(), # remove ticks/text on labels
        axis.ticks.x.top = element_blank(),
        axis.text.y.right = element_blank(),
        axis.ticks.y.right = element_blank(),
        axis.title.x.bottom = element_blank(), # remove titles
        axis.title.y.left = element_blank())

training_75 = plot_grid(
  task_1 + ggtitle("") +
    theme(legend.position = "none", axis.title.x=element_blank(), plot.title = element_blank(),
          axis.ticks.x = element_blank(), axis.text.x = element_blank()),
  task_2 + ggtitle("") +
    theme(legend.position = "none", axis.title.x=element_blank(),axis.title.y=element_blank(),
          plot.title = element_blank(), axis.ticks.x = element_blank(), axis.text.x = element_blank()),
  task_3 + ggtitle("") +
    theme(legend.position = "none", axis.title.x=element_blank(),axis.title.y=element_blank(),
          plot.title = element_blank(), axis.ticks.x = element_blank(), axis.text.x = element_blank()),
  task_4 + ggtitle("") +
    theme(legend.position = "none", axis.title.x=element_blank(),
          plot.title = element_blank(), axis.ticks.x = element_blank(), axis.text.x = element_blank()),
  ncol=4,
  rel_widths =  c(1,1,1,1.1),
  label_size = 17
)
