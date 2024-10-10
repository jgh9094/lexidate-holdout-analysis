# split == 75
# classification

# plot for task 167104
task_1 = filter(classification_scores, task_id == c_task_id_lists[1] & split == "75%") %>%
  ggpaired(.,
           color = 'black',
           cond1 = "tournament",
           cond2 = "lexicase",
           fill = "condition",
           palette = "rickandmorty",
           point.size = 2.0,
           width = 0.1,
           ggtheme = theme_gray()) + p_theme +
  scale_y_continuous(
    name="Accuracy%",
    labels = scales::percent,
  ) +
  scale_x_discrete(
    name="Treatment"
  ) + ggtitle('Task 146818')


# plot for task 168784
task_2 = filter(classification_scores, task_id == c_task_id_lists[2] & split == "75%") %>%
  ggpaired(.,
           color = 'black',
           cond1 = "tournament",
           cond2 = "lexicase",
           fill = "condition",
           palette = "rickandmorty",
           point.size = 2.0,
           width = 0.1,
           ggtheme = theme_gray()) + p_theme +
  scale_y_continuous(
    name="Accuracy%",
    labels = scales::percent,
  ) +
  scale_x_discrete(
    name="Treatment"
  ) + ggtitle('Task 168784')


# plot for task 190137
task_3 = filter(classification_scores, task_id == c_task_id_lists[3] & split == "75%") %>%
  ggpaired(.,
           color = 'black',
           cond1 = "tournament",
           cond2 = "lexicase",
           fill = "condition",
           palette = "rickandmorty",
           point.size = 2.0,
           width = 0.1,
           ggtheme = theme_gray()) + p_theme +
  scale_y_continuous(
    name="Accuracy%",
    labels = scales::percent,
  ) +
  scale_x_discrete(
    name="Treatment"
  ) + ggtitle('Task 190137')


# plot for task 359969
task_4 = filter(classification_scores, task_id == c_task_id_lists[4] & split == "75%") %>%
  ggpaired(.,
           color = 'black',
           cond1 = "tournament",
           cond2 = "lexicase",
           fill = "condition",
           palette = "rickandmorty",
           point.size = 2.0,
           width = 0.1,
           ggtheme = theme_gray()) + p_theme +
  scale_y_continuous(
    name="Split of 75%",
    labels = scales::percent,
    sec.axis = dup_axis()
  ) +
  scale_x_discrete(
    name="Treatment"
  ) + ggtitle('Task 359969') +
  p_theme +
  theme(plot.title = element_text(hjust=0.5),
        axis.text.x.top = element_blank(), # remove ticks/text on labels
        axis.ticks.x.top = element_blank(),
        axis.text.y.right = element_blank(),
        axis.ticks.y.right = element_blank(),
        axis.title.x.bottom = element_blank(), # remove titles
        axis.title.y.left = element_blank())

testing_75 = plot_grid(
  task_1 + ggtitle("") +
    theme(legend.position = "none", axis.title.x=element_blank(),
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
